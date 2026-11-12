//! Integration: `Config::accept_plain_rtp_offers` lets an endpoint that
//! requires secure media answer one inbound plain-RTP offer in kind.
//!
//! This is the answering counterpart of `with_plain_rtp`, and what a B2BUA
//! needs for calls that *arrive* from a plain RTP peer (a SIP carrier trunk, a
//! desk phone) while its other legs stay on the secure profile. The exemption
//! is chosen by the offer: an offer that carries keying is still negotiated
//! securely, and without the option a plain offer is still refused.

use std::time::Duration;

use rvoip_sip::api::events::Event;
use rvoip_sip::api::unified::{Config, UnifiedCoordinator};
use rvoip_sip::SrtpKeyingMode;

fn reserve_loopback_port() -> u16 {
    std::net::UdpSocket::bind("127.0.0.1:0")
        .expect("reserve loopback port")
        .local_addr()
        .expect("local addr")
        .port()
}

fn two_ports() -> (u16, u16) {
    let first = reserve_loopback_port();
    let mut second = reserve_loopback_port();
    while second == first {
        second = reserve_loopback_port();
    }
    (first, second)
}

async fn wait_for<T>(
    events: &mut rvoip_sip::api::stream_peer::EventReceiver,
    timeout: Duration,
    mut predicate: impl FnMut(&Event) -> Option<T>,
) -> Option<T> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return None;
        }
        match tokio::time::timeout(remaining, events.next()).await {
            Ok(Some(event)) => {
                if let Some(value) = predicate(&event) {
                    return Some(value);
                }
            }
            Ok(None) | Err(_) => return None,
        }
    }
}

/// A secure-voice endpoint: SRTP offered and required (SDES, so the tests run
/// with default features).
fn secure_config(name: &str, port: u16, accept_plain: bool) -> Config {
    let mut config = Config::local(name, port);
    config.offer_srtp = true;
    config.srtp_required = true;
    config.srtp_keying = SrtpKeyingMode::Sdes;
    config.accept_plain_rtp_offers = accept_plain;
    config
}

/// Places a call from `caller` to a callee built from `callee_cfg`, has the
/// callee accept it, and returns the caller's outcome: the answer SDP, the
/// failure status, or `Err(0)` when the call was neither answered nor failed
/// within the wait (a refused offer is not always signalled to the caller).
async fn call(caller_cfg: Config, callee_cfg: Config, callee_port: u16) -> Result<String, u16> {
    let caller = UnifiedCoordinator::new(caller_cfg).await.expect("caller");
    let callee = UnifiedCoordinator::new(callee_cfg).await.expect("callee");
    let mut caller_events = caller.events().await.expect("caller events");
    let mut callee_events = callee.events().await.expect("callee events");
    tokio::time::sleep(Duration::from_millis(100)).await;

    caller
        .invite(
            Some("sip:caller@127.0.0.1".to_string()),
            format!("sip:callee@127.0.0.1:{callee_port}"),
        )
        .send()
        .await
        .expect("invite.send()");

    let incoming = wait_for(
        &mut callee_events,
        Duration::from_secs(8),
        |event| match event {
            Event::IncomingCall { call_id, .. } => Some(call_id.clone()),
            _ => None,
        },
    )
    .await
    .expect("callee did not see the INVITE");
    // Accepting is where the offer is answered; a refused offer surfaces to
    // the caller as a failed call, so the accept result itself is not asserted.
    let _ = callee.accept_call(&incoming).await;

    let outcome = wait_for(
        &mut caller_events,
        Duration::from_secs(5),
        |event| match event {
            Event::CallAnswered { call_id, sdp, .. } => {
                Some(Ok((call_id.clone(), sdp.clone().unwrap_or_default())))
            }
            Event::CallFailed { status_code, .. } => Some(Err(*status_code)),
            _ => None,
        },
    )
    .await
    .unwrap_or(Err(0));

    match outcome {
        Ok((session, sdp)) => {
            let _ = caller.hangup(&session).await;
            Ok(sdp)
        }
        Err(status) => Err(status),
    }
}

fn media_line(sdp: &str) -> &str {
    sdp.lines()
        .find(|line| line.starts_with("m=audio"))
        .unwrap_or_else(|| panic!("no m=audio line in\n{sdp}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn secure_endpoint_answers_a_plain_rtp_offer_when_told_to() {
    let _ = tracing_subscriber::fmt::try_init();
    let (caller_port, callee_port) = two_ports();

    let answer = call(
        Config::local("caller", caller_port),
        secure_config("callee", callee_port, true),
        callee_port,
    )
    .await
    .unwrap_or_else(|status| panic!("plain-RTP call was refused with {status}"));

    let line = media_line(&answer);
    assert!(
        line.contains(" RTP/AVP "),
        "a plain-RTP offer must be answered with RTP/AVP, got `{line}`\n{answer}"
    );
    assert!(
        !answer.contains("a=crypto:") && !answer.contains("a=fingerprint:"),
        "a plain-RTP answer must carry no keying\n{answer}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plain_rtp_offer_is_still_refused_without_the_option() {
    let _ = tracing_subscriber::fmt::try_init();
    let (caller_port, callee_port) = two_ports();

    let outcome = call(
        Config::local("caller", caller_port),
        secure_config("callee", callee_port, false),
        callee_port,
    )
    .await;

    assert!(
        outcome.is_err(),
        "srtp_required must still refuse a plain-RTP offer by default, but it was answered:\n{}",
        outcome.unwrap_or_default()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_secure_offer_is_not_downgraded_by_the_option() {
    let _ = tracing_subscriber::fmt::try_init();
    let (caller_port, callee_port) = two_ports();

    let answer = call(
        secure_config("caller", caller_port, false),
        secure_config("callee", callee_port, true),
        callee_port,
    )
    .await
    .unwrap_or_else(|status| panic!("secure call was refused with {status}"));

    let line = media_line(&answer);
    assert!(
        line.contains(" RTP/SAVP "),
        "an SRTP offer must still be answered with RTP/SAVP, got `{line}`\n{answer}"
    );
    assert!(
        answer.contains("a=crypto:"),
        "the SRTP answer lost its keying\n{answer}"
    );
}

/// The WebRTC B2BUA shape: DTLS-SRTP keying, secure media and RTCP-mux both
/// required. A carrier's or desk phone's plain offer (no fingerprint, no
/// `a=rtcp-mux`) must still be answerable on that endpoint.
#[cfg(feature = "dtls-srtp")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dtls_srtp_endpoint_answers_a_plain_rtp_offer_when_told_to() {
    let _ = tracing_subscriber::fmt::try_init();
    let (caller_port, callee_port) = two_ports();

    let mut callee = Config::local("callee", callee_port);
    callee.offer_srtp = true;
    callee.srtp_required = true;
    callee.srtp_keying = SrtpKeyingMode::DtlsSrtp;
    callee.rtcp_mux_required = true;
    callee.accept_plain_rtp_offers = true;

    // A plain peer that does not multiplex RTCP either.
    let mut caller = Config::local("caller", caller_port);
    caller.offer_rtcp_mux = false;

    let answer = call(caller, callee, callee_port)
        .await
        .unwrap_or_else(|status| panic!("plain-RTP call was refused with {status}"));

    let line = media_line(&answer);
    assert!(
        line.contains(" RTP/AVP "),
        "a plain-RTP offer must be answered with RTP/AVP, got `{line}`\n{answer}"
    );
    assert!(
        !answer.contains("a=fingerprint:") && !answer.contains("a=candidate:"),
        "a plain-RTP answer must carry no DTLS fingerprint or ICE candidates\n{answer}"
    );
}
