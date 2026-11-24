//! Real-socket checks that a leg's outgoing RTP stream keeps one timeline
//! while its producer changes: the application's own audio, a relayed or
//! transcoded peer, a different peer after the bridge is replaced, DTMF.
//!
//! Receivers play audio in timestamp order and drop a packet whose timestamp
//! is not newer than audio they already played, so the stream a leg sends
//! must never step back when its producer changes, however unrelated the
//! producers' clocks are.

use super::{BridgeMode, MediaConfig, MediaSessionController};
use crate::types::{AudioFrame, DialogId};
use bytes::Bytes;
use rvoip_rtp_core::packet::rtp::RtpPacket;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const PCMU: u8 = 0;
const PCMA: u8 = 8;
const TELEPHONE_EVENT: u8 = 101;
const FRAME: u32 = 160;
const PACE: Duration = Duration::from_millis(20);
/// Payload byte the first far party sends, so its packets can be told apart
/// on the observed leg.
const FAR_A: u8 = 0x11;
const FAR_B: u8 = 0x22;

/// A media session and the socket standing in for its remote party.
struct Leg {
    id: DialogId,
    peer: Arc<UdpSocket>,
    port: u16,
}

async fn leg(controller: &MediaSessionController, name: &str, codec: &str) -> Leg {
    let peer = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    let id = DialogId::new(name);
    controller
        .start_media(
            id.clone(),
            MediaConfig {
                local_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
                remote_addr: Some(peer.local_addr().unwrap()),
                preferred_codec: Some(codec.to_string()),
                parameters: Default::default(),
            },
        )
        .await
        .unwrap();
    let port = controller
        .get_session_info(&id)
        .await
        .unwrap()
        .rtp_port
        .unwrap();
    Leg { id, peer, port }
}

/// Record every RTP packet the leg's remote party receives until stopped.
fn observe(
    leg: &Leg,
    payload_types: &'static [u8],
) -> (oneshot::Sender<()>, JoinHandle<Vec<RtpPacket>>) {
    let (stop, mut stopped) = oneshot::channel();
    let peer = Arc::clone(&leg.peer);
    let task = tokio::spawn(async move {
        let mut packets = Vec::new();
        let mut buf = vec![0u8; 2048];
        loop {
            tokio::select! {
                _ = &mut stopped => break,
                received = peer.recv_from(&mut buf) => {
                    let (len, _) = received.unwrap();
                    // RTCP shares the socket when multiplexed.
                    if let Ok(packet) = RtpPacket::parse(&buf[..len]) {
                        if payload_types.contains(&packet.header.payload_type) {
                            packets.push(packet);
                        }
                    }
                }
            }
        }
        packets
    });
    (stop, task)
}

async fn finish(observer: (oneshot::Sender<()>, JoinHandle<Vec<RtpPacket>>)) -> Vec<RtpPacket> {
    tokio::time::sleep(Duration::from_millis(150)).await;
    let _ = observer.0.send(());
    observer.1.await.unwrap()
}

/// The application's own audio: `frames` 20 ms frames stamped from its own
/// clock starting at `first`, paced in real time.
async fn app_audio(controller: &MediaSessionController, leg: &Leg, first: u32, frames: u32) {
    for i in 0..frames {
        let frame = AudioFrame::new(
            vec![3_000; FRAME as usize],
            8_000,
            1,
            first.wrapping_add(i * FRAME),
        );
        controller
            .encode_and_send_audio(&leg.id, frame)
            .await
            .unwrap();
        tokio::time::sleep(PACE).await;
    }
}

/// A far party sending 20 ms frames into `leg` from its own clock and SSRC.
/// Frames listed in `lost` are skipped, as if lost on the way.
struct Far {
    ssrc: u32,
    sequence: u16,
    timestamp: u32,
    pt: u8,
    fill: u8,
}

impl Far {
    async fn send(&mut self, leg: &Leg, frames: u32, lost: &[u32]) {
        for i in 0..frames {
            if !lost.contains(&i) {
                let packet = RtpPacket::new_with_payload(
                    self.pt,
                    self.sequence,
                    self.timestamp,
                    self.ssrc,
                    Bytes::from(vec![self.fill; FRAME as usize]),
                );
                leg.peer
                    .send_to(&packet.serialize().unwrap(), ("127.0.0.1", leg.port))
                    .await
                    .unwrap();
            }
            self.sequence = self.sequence.wrapping_add(1);
            self.timestamp = self.timestamp.wrapping_add(FRAME);
            tokio::time::sleep(PACE).await;
        }
    }
}

fn at_or_after(a: u32, b: u32) -> bool {
    a.wrapping_sub(b) < 1 << 31
}

/// One SSRC, contiguous sequence numbers, and timestamps that never go
/// back, across every packet the leg sent.
///
/// RFC 4733 events are the one allowed exception: every packet of an event
/// repeats the event's start timestamp while audio around it moves on. So
/// an event's later packets are exempt, but its start may not be before
/// anything sent earlier, and audio after it may not be before its start.
fn assert_one_timeline(packets: &[RtpPacket]) {
    assert!(!packets.is_empty());
    let mut newest: Option<u32> = None;
    let mut events_started: Vec<u32> = Vec::new();
    for (index, packet) in packets.iter().enumerate() {
        let header = &packet.header;
        if let Some(before) = index.checked_sub(1).map(|i| &packets[i].header) {
            assert_eq!(
                header.sequence_number,
                before.sequence_number.wrapping_add(1),
                "sequence gap"
            );
            assert_eq!(header.ssrc, before.ssrc, "one outgoing SSRC");
        }
        let repeat =
            header.payload_type == TELEPHONE_EVENT && events_started.contains(&header.timestamp);
        if repeat {
            continue;
        }
        if let Some(newest) = newest {
            assert!(
                at_or_after(header.timestamp, newest),
                "timestamp went back from {} to {} (pt {}) at packet {}: {:?}",
                newest,
                header.timestamp,
                header.payload_type,
                index,
                summary(packets)
            );
        }
        if header.payload_type == TELEPHONE_EVENT {
            events_started.push(header.timestamp);
        }
        newest = Some(header.timestamp);
    }
}

fn summary(packets: &[RtpPacket]) -> Vec<(u8, u32, bool, u8)> {
    packets
        .iter()
        .map(|p| {
            (
                p.header.payload_type,
                p.header.timestamp,
                p.header.marker,
                p.payload.first().copied().unwrap_or_default(),
            )
        })
        .collect()
}

/// Split audio packets into runs by who produced them.
fn runs(packets: &[RtpPacket]) -> Vec<(u8, Vec<&RtpPacket>)> {
    let mut runs: Vec<(u8, Vec<&RtpPacket>)> = Vec::new();
    for packet in packets {
        let who = match packet.payload.first() {
            Some(&FAR_A) => FAR_A,
            Some(&FAR_B) => FAR_B,
            _ => 0,
        };
        match runs.last_mut() {
            Some((last, run)) if *last == who => run.push(packet),
            _ => runs.push((who, vec![packet])),
        }
    }
    runs
}

fn steps(run: &[&RtpPacket]) -> Vec<u32> {
    run.windows(2)
        .map(|pair| {
            pair[1]
                .header
                .timestamp
                .wrapping_sub(pair[0].header.timestamp)
        })
        .collect()
}

/// Prompt, then a relayed party, then a prompt again: the stream goes on
/// forward at both switches, each switch is marked, and the relayed audio
/// keeps its own spacing (including a lost packet's gap).
#[tokio::test]
async fn app_audio_then_relayed_audio_then_app_audio_stay_on_one_timeline() {
    let controller = MediaSessionController::new();
    let observed = leg(&controller, "timeline-app-relay-app", "PCMU").await;
    let far = leg(&controller, "timeline-app-relay-far", "PCMU").await;
    let observer = observe(&observed, &[PCMU]);

    // The application stamps from a clock of its own...
    app_audio(&controller, &observed, 1_000_000, 8).await;
    // ...the far party from an unrelated one, behind it on the wrap.
    let bridge = controller
        .bridge_sessions(far.id.clone(), observed.id.clone())
        .await
        .unwrap();
    let mut a = Far {
        ssrc: 0xA0A0_A0A0,
        sequence: 100,
        timestamp: 3_000_000_000,
        pt: PCMU,
        fill: FAR_A,
    };
    a.send(&far, 10, &[5]).await;
    tokio::time::sleep(Duration::from_millis(60)).await;
    drop(bridge);
    tokio::time::sleep(Duration::from_millis(40)).await;
    // The application goes on from its own clock, unaware of the bridge.
    app_audio(&controller, &observed, 1_000_000 + 8 * FRAME, 8).await;

    let packets = finish(observer).await;
    assert_one_timeline(&packets);
    let runs = runs(&packets);
    let shape: Vec<u8> = runs.iter().map(|(who, _)| *who).collect();
    assert_eq!(shape, vec![0, FAR_A, 0], "{:?}", summary(&packets));
    for (_, run) in &runs {
        assert!(run[0].header.marker, "a switch starts with a marker");
        assert!(run[1..].iter().all(|p| !p.header.marker));
    }
    // The relayed party's spacing is its own: 20 ms steps and one lost frame.
    let relayed = steps(&runs[1].1);
    assert_eq!(relayed.iter().filter(|s| **s == 2 * FRAME).count(), 1);
    assert!(relayed.iter().all(|s| *s == FRAME || *s == 2 * FRAME));
    // The application's own audio before the bridge is sent as stamped.
    assert_eq!(runs[0].1[0].header.timestamp, 1_000_000);
}

/// A bridge to one party, then to another with an unrelated clock behind the
/// first: the stream goes on forward and the switch is marked.
#[tokio::test]
async fn relayed_party_a_then_party_b_stay_on_one_timeline() {
    let controller = MediaSessionController::new();
    let observed = leg(&controller, "timeline-ab-observed", "PCMU").await;
    let far_a = leg(&controller, "timeline-ab-a", "PCMU").await;
    let far_b = leg(&controller, "timeline-ab-b", "PCMU").await;
    let observer = observe(&observed, &[PCMU]);

    let first = controller
        .bridge_sessions(far_a.id.clone(), observed.id.clone())
        .await
        .unwrap();
    let mut a = Far {
        ssrc: 0xAAAA_0001,
        sequence: 1,
        timestamp: 3_000_000_000,
        pt: PCMU,
        fill: FAR_A,
    };
    a.send(&far_a, 8, &[]).await;
    tokio::time::sleep(Duration::from_millis(40)).await;
    drop(first);
    tokio::time::sleep(Duration::from_millis(20)).await;
    let second = controller
        .bridge_sessions(far_b.id.clone(), observed.id.clone())
        .await
        .unwrap();
    let mut b = Far {
        ssrc: 0xBBBB_0002,
        sequence: 40_000,
        timestamp: 2_000_000_000,
        pt: PCMU,
        fill: FAR_B,
    };
    b.send(&far_b, 8, &[]).await;

    let packets = finish(observer).await;
    drop(second);
    assert_one_timeline(&packets);
    let runs = runs(&packets);
    let shape: Vec<u8> = runs.iter().map(|(who, _)| *who).collect();
    assert_eq!(shape, vec![FAR_A, FAR_B], "{:?}", summary(&packets));
    assert!(
        runs[1].1[0].header.marker,
        "the switch starts with a marker"
    );
    for (_, run) in &runs {
        assert!(steps(run).iter().all(|s| *s == FRAME));
    }
    // The wall time between the parties is in the timeline too.
    let gap = runs[1].1[0]
        .header
        .timestamp
        .wrapping_sub(runs[0].1.last().unwrap().header.timestamp);
    assert!((FRAME..=FRAME * 20).contains(&gap), "gap {gap}");
}

/// The transcode path re-encodes on its own clock, which started from the
/// far party's timestamps: the leg's prompt before and after it and the
/// transcoded audio all stay on one timeline.
#[tokio::test]
async fn app_audio_then_transcoded_audio_then_app_audio_stay_on_one_timeline() {
    let controller = MediaSessionController::new();
    let observed = leg(&controller, "timeline-xcode-observed", "PCMA").await;
    let far = leg(&controller, "timeline-xcode-far", "PCMU").await;
    let observer = observe(&observed, &[PCMA]);

    app_audio(&controller, &observed, 1_000_000, 8).await;
    let bridge = controller
        .bridge_sessions_with_mode(far.id.clone(), observed.id.clone(), BridgeMode::Transcode)
        .await
        .unwrap();
    let mut a = Far {
        ssrc: 0xC0C0_0003,
        sequence: 7,
        timestamp: 3_000_000_000,
        pt: PCMU,
        fill: FAR_A,
    };
    a.send(&far, 15, &[]).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    drop(bridge);
    tokio::time::sleep(Duration::from_millis(40)).await;
    app_audio(&controller, &observed, 1_000_000 + 8 * FRAME, 8).await;

    let packets = finish(observer).await;
    assert_one_timeline(&packets);
    // Prompt, transcoded audio, prompt: at least two producer switches, and
    // each is marked.
    let switches = packets.iter().skip(1).filter(|p| p.header.marker).count();
    assert!(switches >= 2, "{:?}", summary(&packets));
    assert!(packets.len() >= 8 + 8 + 5, "{:?}", summary(&packets));
}

/// DTMF from the node in the middle of its own audio and in the middle of
/// relayed audio: every packet of an event carries one timestamp, an event
/// never starts before the audio already sent, and the audio after it never
/// goes back before the event.
#[tokio::test]
async fn dtmf_in_the_middle_of_app_and_relayed_audio_stays_on_one_timeline() {
    let controller = MediaSessionController::new();
    let observed = leg(&controller, "timeline-dtmf-observed", "PCMU").await;
    let far = leg(&controller, "timeline-dtmf-far", "PCMU").await;
    let observer = observe(&observed, &[PCMU, TELEPHONE_EVENT]);

    app_audio(&controller, &observed, 1_000_000, 3).await;
    controller
        .send_dtmf_packet(&observed.id, '4', 100)
        .await
        .unwrap();
    app_audio(&controller, &observed, 1_000_000 + 3 * FRAME, 8).await;
    let bridge = controller
        .bridge_sessions(far.id.clone(), observed.id.clone())
        .await
        .unwrap();
    let mut a = Far {
        ssrc: 0xD0D0_0004,
        sequence: 500,
        timestamp: 3_000_000_000,
        pt: PCMU,
        fill: FAR_A,
    };
    a.send(&far, 4, &[]).await;
    controller
        .send_dtmf_packet(&observed.id, '7', 100)
        .await
        .unwrap();
    a.send(&far, 10, &[]).await;
    drop(bridge);

    let packets = finish(observer).await;
    assert_one_timeline(&packets);
    for digit in [4u8, 7] {
        let event: Vec<u32> = packets
            .iter()
            .filter(|p| p.header.payload_type == TELEPHONE_EVENT && p.payload[0] == digit)
            .map(|p| p.header.timestamp)
            .collect();
        assert!(
            event.len() >= 3,
            "digit {digit} was sent: {:?}",
            summary(&packets)
        );
        assert!(
            event.iter().all(|ts| *ts == event[0]),
            "one timestamp per event: {event:?}"
        );
    }
    // Each event starts on the audio clock: within one frame of the audio
    // packet sent just before it.
    for (index, packet) in packets.iter().enumerate() {
        if packet.header.payload_type == TELEPHONE_EVENT && packet.header.marker {
            let before = packets[..index]
                .iter()
                .rev()
                .find(|p| p.header.payload_type == PCMU)
                .unwrap();
            let step = packet
                .header
                .timestamp
                .wrapping_sub(before.header.timestamp);
            assert!(step <= FRAME, "event starts {step} after the audio");
        }
    }
}

/// An application that stamps its audio from the leg's next timestamp is
/// sent with exactly the timestamps it stamped, right after a bridge.
#[tokio::test]
async fn audio_stamped_from_next_rtp_timestamp_is_sent_as_stamped() {
    let controller = MediaSessionController::new();
    let observed = leg(&controller, "timeline-next-observed", "PCMU").await;
    let far = leg(&controller, "timeline-next-far", "PCMU").await;
    let observer = observe(&observed, &[PCMU]);

    assert_eq!(controller.next_rtp_timestamp(&observed.id).await, None);
    let bridge = controller
        .bridge_sessions(far.id.clone(), observed.id.clone())
        .await
        .unwrap();
    let mut a = Far {
        ssrc: 0xE0E0_0005,
        sequence: 9,
        timestamp: 123_456_789,
        pt: PCMU,
        fill: FAR_A,
    };
    a.send(&far, 6, &[]).await;
    drop(bridge);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let next = controller.next_rtp_timestamp(&observed.id).await.unwrap();
    app_audio(&controller, &observed, next, 6).await;

    let packets = finish(observer).await;
    assert_one_timeline(&packets);
    let runs = runs(&packets);
    assert_eq!(runs.len(), 2, "{:?}", summary(&packets));
    let stamped: Vec<u32> = runs[1].1.iter().map(|p| p.header.timestamp).collect();
    let expected: Vec<u32> = (0..6).map(|i| next.wrapping_add(i * FRAME)).collect();
    assert_eq!(stamped, expected);
    // The 100 ms pause after the bridge is in the timeline.
    let gap = next.wrapping_sub(runs[0].1.last().unwrap().header.timestamp);
    assert!(gap >= 800, "gap {gap}");
}
