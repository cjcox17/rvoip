//! End-to-end audio and RTP clock checks for the Opus-to-PSTN bridge.

use super::{BridgeMode, MediaConfig, MediaSessionController};
use crate::codec::audio::{AudioCodec, G711Codec, OpusCodec, OpusConfig};
use crate::processing::format::converter::{ConversionParams, FormatConverter};
use crate::types::{AudioFrame, DialogId, SampleRate};
use bytes::Bytes;
use rvoip_rtp_core::packet::rtp::RtpPacket;
use std::f64::consts::TAU;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::time::timeout;

const OPUS_PT: u8 = 112;
const FRAMES: usize = 14;
const FIRST_TIMESTAMP: u32 = u32::MAX - 5 * 960;
const FIRST_SEQUENCE: u16 = u16::MAX - 5;

fn timestamp(index: usize) -> u32 {
    FIRST_TIMESTAMP.wrapping_add(index as u32 * 960)
}

/// Encode one continuous narrowband chirp in stereo. The independently decoded
/// and downmixed reference uses each encoded packet exactly once, in order.
fn packets_and_reference() -> (Vec<Bytes>, Vec<AudioFrame>) {
    let opus = || OpusCodec::new(SampleRate::Rate48000, 2, OpusConfig::default()).unwrap();
    let mut encoder = opus();
    let mut decoder = opus();
    let mut converter = FormatConverter::new();
    let target = ConversionParams::new(SampleRate::Rate8000, 1);
    let mut packets = Vec::new();
    let mut reference = Vec::new();
    for index in 0..FRAMES {
        let samples = (0..960)
            .flat_map(|i| {
                let t = (960 * index + i) as f64 / 48_000.0;
                let chirp = (TAU * (320.0 * t + 1900.0 * t * t)).sin();
                [(10_000.0 * chirp) as i16, (6_000.0 * chirp) as i16]
            })
            .collect();
        let frame = AudioFrame::new(samples, 48_000, 2, timestamp(index));
        let payload = Bytes::from(encoder.encode(&frame).unwrap());
        let mut decoded = decoder.decode(&payload).unwrap();
        decoded.timestamp = frame.timestamp;
        let converted = converter.convert_frame(&decoded, &target).unwrap().frame;
        assert!(converted
            .samples
            .iter()
            .any(|sample| sample.unsigned_abs() > 500));
        reference.push(converted);
        packets.push(payload);
    }
    (packets, reference)
}

fn config(peer: SocketAddr, name: &str, pt: u8, clock: u32, channels: u8) -> MediaConfig {
    MediaConfig {
        local_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
        remote_addr: Some(peer),
        preferred_codec: None,
        parameters: Default::default(),
    }
    .with_negotiated_audio_codec(name, pt, clock, channels)
}

struct Phase<'a> {
    name: &'a str,
    pt: u8,
    arrival_order: &'a [usize],
}

async fn assert_transcoded_audio(test_name: &str, phases: &[Phase<'_>]) {
    let first = &phases[0];
    let source = DialogId::new(format!("opus-to-{test_name}"));
    let destination = DialogId::new(format!("{test_name}-peer"));
    let source_peer = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let destination_peer = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let controller = MediaSessionController::new();
    controller
        .start_media(
            source.clone(),
            config(
                source_peer.local_addr().unwrap(),
                "opus",
                OPUS_PT,
                48_000,
                2,
            ),
        )
        .await
        .unwrap();
    controller
        .start_media(
            destination.clone(),
            config(
                destination_peer.local_addr().unwrap(),
                first.name,
                first.pt,
                8_000,
                1,
            ),
        )
        .await
        .unwrap();
    let port = controller
        .get_session_info(&source)
        .await
        .unwrap()
        .rtp_port
        .unwrap();
    let bridge = controller
        .bridge_sessions_with_mode(source.clone(), destination.clone(), BridgeMode::Transcode)
        .await
        .unwrap();

    let (app_tx, app_rx) = tokio::sync::mpsc::channel(32);
    controller
        .set_audio_frame_callback(source.clone(), app_tx)
        .await
        .unwrap();
    let mut app_rx = Some(app_rx);
    let (packets, reference) = packets_and_reference();
    let mut buf = [0u8; 2048];
    let mut previous_sequence: Option<u16> = None;
    let mut first_timestamp = None;
    let mut next_index = 0;
    for (phase_index, phase) in phases.iter().enumerate() {
        let Phase {
            name,
            pt,
            arrival_order,
        } = *phase;
        if phase_index != 0 {
            let mut rx = app_rx.take().unwrap();
            let received = timeout(Duration::from_secs(2), rx.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(received.timestamp, timestamp(0));
            drop(rx);
            controller
                .update_media(
                    destination.clone(),
                    config(destination_peer.local_addr().unwrap(), name, pt, 8_000, 1),
                )
                .await
                .unwrap();
        }
        let mut g711 = if pt == 0 {
            G711Codec::mu_law(8_000, 1)
        } else {
            G711Codec::a_law(8_000, 1)
        }
        .unwrap();
        for &index in arrival_order {
            let packet = RtpPacket::new_with_payload(
                OPUS_PT,
                FIRST_SEQUENCE.wrapping_add(index as u16),
                timestamp(index),
                0x12345678,
                packets[index].clone(),
            );
            source_peer
                .send_to(&packet.serialize().unwrap(), ("127.0.0.1", port))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(6)).await;
        }
        let end = *arrival_order.iter().max().unwrap() + 1;
        for (index, frame) in reference.iter().enumerate().take(end).skip(next_index) {
            let expected = g711.encode(frame).unwrap();
            let len = timeout(Duration::from_secs(2), destination_peer.recv(&mut buf))
                .await
                .expect("missing transcoded audio")
                .unwrap();
            let packet = RtpPacket::parse(&buf[..len]).unwrap();
            assert_eq!(packet.header.payload_type, pt);
            assert_eq!(packet.payload.len(), 160);
            let anchor = *first_timestamp.get_or_insert(packet.header.timestamp);
            assert_eq!(
                packet.header.timestamp,
                anchor.wrapping_add(index as u32 * 160),
                "{name} packet {index} has a gap or duplicate"
            );
            if let Some(seq) = previous_sequence {
                assert_eq!(packet.header.sequence_number, seq.wrapping_add(1));
            }
            previous_sequence = Some(packet.header.sequence_number);
            assert!(packet.payload.as_ref() == expected, "{name} packet {index} differs from the sequential Opus decode / downmix / G.711 reference");
        }
        next_index = end;
    }
    assert_eq!(next_index, FRAMES);
    // This PT belongs to the other leg, not this Opus source. A stray or
    // stale audio PT must never bypass the decoder into the PSTN stream.
    let stray = RtpPacket::new_with_payload(
        phases.last().unwrap().pt,
        FIRST_SEQUENCE.wrapping_add(FRAMES as u16),
        timestamp(FRAMES),
        0x12345678,
        Bytes::from_static(&[0x55; 160]),
    );
    source_peer
        .send_to(&stray.serialize().unwrap(), ("127.0.0.1", port))
        .await
        .unwrap();

    assert!(
        timeout(Duration::from_millis(100), destination_peer.recv(&mut buf))
            .await
            .is_err(),
        "unexpected duplicate output"
    );
    drop(bridge);
    controller.stop_media(&source).await.unwrap();
    controller.stop_media(&destination).await.unwrap();
}

#[tokio::test]
async fn opus_to_pcmu_restores_order_before_decoding() {
    assert_transcoded_audio(
        "reorder",
        &[Phase {
            name: "PCMU",
            pt: 0,
            arrival_order: &[0, 1, 2, 4, 3, 5, 7, 6, 8, 9, 10, 11, 12, 13],
        }],
    )
    .await;
}

#[tokio::test]
async fn opus_to_pcma_restores_order_and_ignores_duplicates_without_advancing_decoder() {
    assert_transcoded_audio(
        "duplicates",
        &[Phase {
            name: "PCMA",
            pt: 8,
            arrival_order: &[0, 1, 2, 4, 4, 3, 3, 5, 7, 6, 8, 9, 10, 11, 12, 13],
        }],
    )
    .await;
}

#[tokio::test]
async fn bridge_adopts_renegotiated_g711_and_outlives_the_application_callback() {
    assert_transcoded_audio(
        "renegotiation",
        &[
            Phase {
                name: "PCMU",
                pt: 0,
                arrival_order: &[0, 1, 2, 4, 3, 5, 6],
            },
            Phase {
                name: "PCMA",
                pt: 8,
                arrival_order: &[7, 8, 10, 9, 11, 12, 13],
            },
        ],
    )
    .await;
}

/// The outgoing leg keeps its own SSRC and sample clock when the source
/// renegotiates its codec and restarts RTP timestamps and sequence numbers.
#[tokio::test]
async fn source_renegotiation_keeps_destination_timestamps_continuous() {
    let src = DialogId::new("changing-source");
    let dst = DialogId::new("stable-pcmu-peer");
    let source = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let peer = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let controller = MediaSessionController::new();
    let source_addr = source.local_addr().unwrap();
    controller
        .start_media(src.clone(), config(source_addr, "opus", OPUS_PT, 48_000, 2))
        .await
        .unwrap();
    controller
        .start_media(
            dst.clone(),
            config(peer.local_addr().unwrap(), "PCMU", 0, 8_000, 1),
        )
        .await
        .unwrap();
    let port = controller
        .get_session_info(&src)
        .await
        .unwrap()
        .rtp_port
        .unwrap();
    let _bridge = controller
        .bridge_sessions_with_mode(src.clone(), dst.clone(), BridgeMode::Transcode)
        .await
        .unwrap();
    let (opus, _) = packets_and_reference();
    let mut anchor = None;
    let mut output_ssrc = None;
    let mut buf = [0u8; 2048];
    for (phase, (name, pt, origin, step)) in [
        ("opus", OPUS_PT, u32::MAX - 400, 960),
        ("PCMA", 8, 724_000, 160),
        ("PCMU", 0, 17, 160),
    ]
    .into_iter()
    .enumerate()
    {
        if phase != 0 {
            controller
                .update_media(src.clone(), config(source_addr, name, pt, 8_000, 1))
                .await
                .unwrap();
        }
        for (index, opus_payload) in opus.iter().take(3).enumerate() {
            let payload = if phase == 0 {
                opus_payload.clone()
            } else {
                Bytes::from(vec![0xaa; 160])
            };
            let packet = RtpPacket::new_with_payload(
                pt,
                (300 * phase + index) as u16,
                origin.wrapping_add(index as u32 * step),
                77,
                payload,
            );
            source
                .send_to(&packet.serialize().unwrap(), ("127.0.0.1", port))
                .await
                .unwrap();
            let len = timeout(Duration::from_secs(2), peer.recv(&mut buf))
                .await
                .unwrap()
                .unwrap();
            let out = RtpPacket::parse(&buf[..len]).unwrap();
            assert_eq!(out.header.payload_type, 0);
            assert_eq!(out.header.ssrc, *output_ssrc.get_or_insert(out.header.ssrc));
            assert_eq!(
                out.header.timestamp,
                anchor
                    .get_or_insert(out.header.timestamp)
                    .wrapping_add((phase * 3 + index) as u32 * 160),
                "phase {phase}, frame {index}"
            );
        }
    }
    controller.stop_media(&src).await.unwrap();
    controller.stop_media(&dst).await.unwrap();
}

/// Two legs that gave Opus different dynamic payload types still relay, and
/// each leg receives the number it negotiated. A payload type the
/// destination never agreed to is dropped rather than forwarded unmapped.
#[tokio::test]
async fn relay_maps_payload_type_to_destination_legs_negotiated_number() {
    let peer_a = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let peer_b = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let controller = MediaSessionController::new();
    let (a, b) = (DialogId::new("relay-pt-a"), DialogId::new("relay-pt-b"));
    controller
        .start_media(
            a.clone(),
            config(peer_a.local_addr().unwrap(), "opus", 111, 48_000, 2),
        )
        .await
        .unwrap();
    controller
        .start_media(
            b.clone(),
            config(peer_b.local_addr().unwrap(), "opus", 96, 48_000, 2),
        )
        .await
        .unwrap();
    let port_a = controller
        .get_session_info(&a)
        .await
        .unwrap()
        .rtp_port
        .unwrap();
    let bridge = controller
        .bridge_sessions(a.clone(), b.clone())
        .await
        .expect("the same codec relays whatever number each leg gave it");

    let (packets, _) = packets_and_reference();
    let mut received = Vec::new();
    let mut bytes = [0u8; 2048];
    for (index, payload) in packets.iter().take(6).enumerate() {
        // Every other packet carries a payload type nobody negotiated.
        for pt in [111u8, 99] {
            let packet = RtpPacket::new_with_payload(
                pt,
                (index * 2) as u16 + u16::from(pt == 99),
                timestamp(index),
                0x0101_0101,
                payload.clone(),
            );
            peer_a
                .send_to(&packet.serialize().unwrap(), ("127.0.0.1", port_a))
                .await
                .unwrap();
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    while let Ok(Ok(len)) = timeout(Duration::from_millis(300), peer_b.recv(&mut bytes)).await {
        if let Ok(packet) = RtpPacket::parse(&bytes[..len]) {
            // RTCP shares the socket when multiplexed.
            if packet.header.payload_type < 72 || packet.header.payload_type > 76 {
                received.push(packet.header.payload_type);
            }
        }
    }
    drop(bridge);
    assert!(received.len() >= 3, "relayed audio arrived: {received:?}");
    assert!(
        received.iter().all(|pt| *pt == 96),
        "leg B receives only its own payload type: {received:?}"
    );
}
