//! RFC 3261 §12.2.2 (and RFC 6141 §3.4): a UAS that answers a re-INVITE
//! with a 2xx replaces the dialog's remote target with the re-INVITE's
//! Contact. A peer that moved to a new connection (a softphone that
//! reconnected its signalling after a network change) announces the new
//! Contact in its re-INVITE; every later request the UAS sends in the
//! dialog — here its BYE — must go there. A re-INVITE the UAS rejects
//! leaves the target unchanged.
//!
//! A raw-UDP caller sets the call up from one socket, re-INVITEs from a
//! second socket with a Contact naming it, and records where rvoip's BYE
//! arrives.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::net::UdpSocket;
use tokio::time::timeout;

use rvoip_sip::{Config, StreamPeer};
use rvoip_sip_core::builder::SimpleRequestBuilder;
use rvoip_sip_core::parser::parse_message;
use rvoip_sip_core::prelude::*;
use rvoip_sip_dialog::transaction::utils::response_builders::create_response;

const WAIT: Duration = Duration::from_secs(10);

async fn recv_message(sock: &UdpSocket, within: Duration) -> Option<Message> {
    let mut buf = vec![0u8; 16384];
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let (n, _) = timeout(remaining, sock.recv_from(&mut buf))
            .await
            .ok()?
            .ok()?;
        if let Ok(message) = parse_message(&buf[..n]) {
            return Some(message);
        }
    }
}

async fn final_response(sock: &UdpSocket) -> Response {
    loop {
        match recv_message(sock, WAIT).await.expect("final response") {
            Message::Response(response) if response.status_code() >= 200 => return response,
            _ => continue,
        }
    }
}

/// Record every request that reaches `sock` and answer it with 200.
async fn record_and_answer(sock: Arc<UdpSocket>, seen: Arc<Mutex<Vec<Request>>>) {
    let mut buf = vec![0u8; 16384];
    loop {
        let Ok((n, from)) = sock.recv_from(&mut buf).await else {
            return;
        };
        let Ok(Message::Request(request)) = parse_message(&buf[..n]) else {
            continue;
        };
        if request.method() == Method::Ack {
            continue;
        }
        let response = create_response(&request, StatusCode::Ok);
        seen.lock().unwrap().push(request);
        let _ = sock
            .send_to(&Message::Response(response).to_bytes(), from)
            .await;
    }
}

fn audio_sdp(rtp_port: u16, version: u32, payload: &str) -> String {
    format!(
        "v=0\r\no=- 0 {version} IN IP4 127.0.0.1\r\ns=-\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\n\
         m=audio {rtp_port} RTP/AVP {payload}\r\n"
    )
}

struct RawCaller {
    peer: StreamPeer,
    rvoip_addr: SocketAddr,
    target: String,
    from_uri: String,
    call_id: String,
    to_tag: String,
    first: Arc<UdpSocket>,
    moved: Arc<UdpSocket>,
    rtp: UdpSocket,
    call: rvoip_sip::SessionHandle,
}

impl RawCaller {
    /// Set a call up from `first`, with `first` as Contact.
    async fn connect(media_base: u16) -> Self {
        let probe = std::net::UdpSocket::bind("127.0.0.1:0").expect("port probe");
        let rvoip_port = probe.local_addr().unwrap().port();
        drop(probe);
        let mut config = Config::local("bob", rvoip_port);
        config.media_port_start = media_base;
        config.media_port_end = media_base + 100;
        let mut peer = StreamPeer::with_config(config).await.expect("peer");
        let rvoip_addr: SocketAddr = format!("127.0.0.1:{rvoip_port}").parse().unwrap();

        let first = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let moved = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let first_addr = first.local_addr().unwrap();
        let rtp = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let from_uri = format!("sip:alice@{first_addr}");
        let target = format!("sip:bob@{rvoip_addr}");
        let call_id = format!("target-refresh-{}", rand::random::<u32>());
        let invite = SimpleRequestBuilder::new(Method::Invite, &target)
            .unwrap()
            .from("Alice", &from_uri, Some("alice-tag"))
            .to("Bob", &target, None)
            .call_id(&call_id)
            .cseq(1)
            .via(&first_addr.to_string(), "UDP", Some("z9hG4bK-refresh-1"))
            .max_forwards(70)
            .contact(&from_uri, None)
            .content_type("application/sdp")
            .body(
                format!(
                    "{}a=rtpmap:0 PCMU/8000\r\na=sendrecv\r\n",
                    audio_sdp(rtp.local_addr().unwrap().port(), 0, "0")
                )
                .into_bytes(),
            )
            .build();
        first
            .send_to(&Message::Request(invite).to_bytes(), rvoip_addr)
            .await
            .unwrap();
        let incoming = timeout(WAIT, peer.wait_for_incoming())
            .await
            .expect("incoming call in time")
            .expect("incoming call");
        let call = incoming.accept().await.expect("accept");
        let ok = final_response(&first).await;
        assert_eq!(ok.status_code(), 200);
        let to_tag = ok
            .to()
            .and_then(|to| to.tag().map(str::to_string))
            .expect("2xx To tag");
        let caller = Self {
            peer,
            rvoip_addr,
            target,
            from_uri,
            call_id,
            to_tag,
            first,
            moved,
            rtp,
            call,
        };
        caller.ack(&caller.first, 1, "z9hG4bK-refresh-ack-1").await;
        let deadline = Instant::now() + WAIT;
        while !caller.call.is_active().await {
            assert!(Instant::now() < deadline, "call never became active");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        caller
    }

    async fn ack(&self, from: &UdpSocket, cseq: u32, branch: &str) {
        let via = from.local_addr().unwrap().to_string();
        let ack = SimpleRequestBuilder::new(Method::Ack, &self.target)
            .unwrap()
            .from("Alice", &self.from_uri, Some("alice-tag"))
            .to("Bob", &self.target, Some(&self.to_tag))
            .call_id(&self.call_id)
            .cseq(cseq)
            .via(&via, "UDP", Some(branch))
            .max_forwards(70)
            .build();
        from.send_to(&Message::Request(ack).to_bytes(), self.rvoip_addr)
            .await
            .unwrap();
    }

    /// Send a re-INVITE from the second socket, Contact naming it, with an
    /// audio offer on `payload`/`rtpmap`. Returns the final status.
    async fn reinvite_from_moved(&self, payload: &str, rtpmap: &str) -> u16 {
        let moved_addr = self.moved.local_addr().unwrap();
        let contact = format!("sip:alice@{moved_addr}");
        let branch = "z9hG4bK-refresh-2";
        let reinvite = SimpleRequestBuilder::new(Method::Invite, &self.target)
            .unwrap()
            .from("Alice", &self.from_uri, Some("alice-tag"))
            .to("Bob", &self.target, Some(&self.to_tag))
            .call_id(&self.call_id)
            .cseq(2)
            .via(&moved_addr.to_string(), "UDP", Some(branch))
            .max_forwards(70)
            .contact(&contact, None)
            .content_type("application/sdp")
            .body(
                format!(
                    "{}a=rtpmap:{payload} {rtpmap}\r\na=sendrecv\r\n",
                    audio_sdp(self.rtp.local_addr().unwrap().port(), 1, payload)
                )
                .into_bytes(),
            )
            .build();
        self.moved
            .send_to(&Message::Request(reinvite).to_bytes(), self.rvoip_addr)
            .await
            .unwrap();
        let response = final_response(&self.moved).await;
        let status = response.status_code();
        if status < 300 {
            self.ack(&self.moved, 2, "z9hG4bK-refresh-ack-2").await;
        } else {
            // RFC 3261 §17.1.1.3: the ACK for a non-2xx shares the INVITE's
            // branch and is absorbed by the server transaction.
            self.ack(&self.moved, 2, branch).await;
        }
        status
    }

    /// Hang up from rvoip and report which socket the BYE reached.
    async fn hang_up_and_locate_bye(self) -> (Vec<Request>, Vec<Request>) {
        let at_first = Arc::new(Mutex::new(Vec::new()));
        let at_moved = Arc::new(Mutex::new(Vec::new()));
        let first_task = tokio::spawn(record_and_answer(
            Arc::clone(&self.first),
            Arc::clone(&at_first),
        ));
        let moved_task = tokio::spawn(record_and_answer(
            Arc::clone(&self.moved),
            Arc::clone(&at_moved),
        ));
        self.call.hangup().await.expect("hangup");
        let deadline = Instant::now() + WAIT;
        loop {
            let byes = |seen: &Arc<Mutex<Vec<Request>>>| {
                seen.lock()
                    .unwrap()
                    .iter()
                    .any(|request| request.method() == Method::Bye)
            };
            if byes(&at_first) || byes(&at_moved) || Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        // Give a misdirected duplicate a moment to show up too.
        tokio::time::sleep(Duration::from_millis(200)).await;
        first_task.abort();
        moved_task.abort();
        let first = at_first.lock().unwrap().clone();
        let moved = at_moved.lock().unwrap().clone();
        self.peer.shutdown().await.expect("shutdown");
        (first, moved)
    }
}

fn methods(requests: &[Request]) -> Vec<Method> {
    requests.iter().map(Request::method).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bye_after_an_accepted_reinvite_goes_to_its_new_contact() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let caller = RawCaller::connect(47_600).await;
    let moved_uri = format!("sip:alice@{}", caller.moved.local_addr().unwrap());
    assert_eq!(caller.reinvite_from_moved("0", "PCMU/8000").await, 200);

    let (at_first, at_moved) = caller.hang_up_and_locate_bye().await;
    assert!(
        methods(&at_moved).contains(&Method::Bye),
        "the BYE must reach the re-INVITE's Contact; first socket saw {:?}, moved socket saw {:?}",
        methods(&at_first),
        methods(&at_moved)
    );
    let bye = at_moved
        .iter()
        .find(|request| request.method() == Method::Bye)
        .unwrap();
    assert_eq!(
        bye.uri().to_string(),
        moved_uri,
        "the BYE's Request-URI is the refreshed remote target"
    );
    assert!(
        at_first.is_empty(),
        "nothing may go to the old Contact: {:?}",
        methods(&at_first)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rejected_reinvite_leaves_the_remote_target_unchanged() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let caller = RawCaller::connect(47_700).await;
    let first_uri = caller.from_uri.clone();
    // An offer with no codec in common is refused (488).
    let status = caller.reinvite_from_moved("98", "NOSUCH/8000").await;
    assert!(status >= 300, "the re-INVITE must be refused, got {status}");

    let (at_first, at_moved) = caller.hang_up_and_locate_bye().await;
    assert!(
        methods(&at_first).contains(&Method::Bye),
        "the BYE must stay on the original Contact; first socket saw {:?}, moved socket saw {:?}",
        methods(&at_first),
        methods(&at_moved)
    );
    let bye = at_first
        .iter()
        .find(|request| request.method() == Method::Bye)
        .unwrap();
    assert_eq!(bye.uri().to_string(), first_uri);
    assert!(
        at_moved.is_empty(),
        "a refused re-INVITE must not move the target: {:?}",
        methods(&at_moved)
    );
}
