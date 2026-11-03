//! Integration: `SessionHandle::negotiated_media` reports the SDP-committed
//! media format at answer time on both the caller (UAC) and callee (UAS) side.
//!
//! The caller offers the default codec set (Opus, PCMA, PCMU); the callee is
//! configured to offer PCMU only. The answer must therefore commit PCMU, and
//! both handles must report codec `PCMU`, payload type 0, 8000 Hz, 1 channel —
//! read from the session store, not inferred from decoded packets.

use std::time::Duration;

use rvoip_media_core::types::AudioFrame;
use rvoip_sip::api::stream_peer::StreamPeer;
use rvoip_sip::api::unified::Config;

const BOB_SIP_PORT: u16 = 35620;
const ALICE_SIP_PORT: u16 = 35621;
const BOB_MEDIA: (u16, u16) = (35630, 35640);
const ALICE_MEDIA: (u16, u16) = (35641, 35651);

fn pcmu_only_config(name: &str, sip_port: u16, media: (u16, u16)) -> Config {
    let mut config = Config::local(name, sip_port).with_media_ports(media.0, media.1);
    // PCMU only, so the negotiated answer cannot be anything else.
    config.offered_codecs = vec![0];
    config
}

fn default_config(name: &str, sip_port: u16, media: (u16, u16)) -> Config {
    Config::local(name, sip_port).with_media_ports(media.0, media.1)
}

fn assert_pcmu(media: &rvoip_sip::NegotiatedMedia, side: &str) {
    assert_eq!(
        media.codec.to_ascii_uppercase(),
        "PCMU",
        "{side}: negotiated codec must be PCMU, got {}",
        media.codec
    );
    assert_eq!(media.payload_type, 0, "{side}: PCMU is payload type 0");
    assert_eq!(media.clock_rate, 8_000, "{side}: PCMU runs at 8 kHz");
    assert_eq!(media.channels, 1, "{side}: PCMU is mono");
    assert!(media.fmtp.is_none(), "{side}: PCMU carries no fmtp");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn negotiated_media_matches_the_sdp_answer_on_both_sides() {
    let _ = tracing_subscriber::fmt::try_init();

    let mut bob = StreamPeer::with_config(pcmu_only_config(
        "negotiated-bob",
        BOB_SIP_PORT,
        BOB_MEDIA,
    ))
    .await
    .expect("bob peer starts (PCMU-only offer)");
    let bob_shutdown = bob.shutdown_handle();

    let mut alice =
        StreamPeer::with_config(default_config("negotiated-alice", ALICE_SIP_PORT, ALICE_MEDIA))
            .await
            .expect("alice peer starts (default offer)");

    // Bob: accept the call and hold the handle.
    let bob_task = tokio::spawn(async move {
        let incoming = tokio::time::timeout(Duration::from_secs(10), bob.wait_for_incoming())
            .await
            .expect("bob timed out waiting for the INVITE")
            .expect("bob received the incoming call");
        let handle = incoming.accept().await.expect("bob accepts the call");

        // The UAS side must already know the negotiated format right after the
        // 200 OK / ACK path completes.
        let media = tokio::time::timeout(
            Duration::from_secs(5),
            handle.wait_for_negotiated_media(None),
        )
        .await
        .expect("bob timed out waiting for the negotiated format")
        .expect("bob's negotiated media is available");
        assert_pcmu(&media, "bob (UAS)");

        // Keep the call up until Alice hangs up.
        let _ = handle.wait_for_end(Some(Duration::from_secs(10))).await;
        media
    });

    // Alice: place the call.
    let call_id = alice
        .invite(format!("sip:bob@127.0.0.1:{BOB_SIP_PORT}"))
        .send()
        .await
        .expect("alice sends the INVITE");

    let handle = tokio::time::timeout(Duration::from_secs(10), alice.wait_for_answered(&call_id))
        .await
        .expect("alice timed out waiting for the answer")
        .expect("alice's call was answered");

    // The UAC side reads the same committed format from its own store.
    let media = tokio::time::timeout(
        Duration::from_secs(5),
        handle.wait_for_negotiated_media(None),
    )
    .await
    .expect("alice timed out waiting for the negotiated format")
    .expect("alice's negotiated media is available");
    assert_pcmu(&media, "alice (UAC)");

    // A plain read (no wait) sees the same value.
    let read_again = handle
        .negotiated_media()
        .await
        .expect("negotiated media read succeeds");
    assert_eq!(read_again.as_ref(), Some(&media));

    // Silence toward Bob keeps the media path alive while the format is being
    // asserted (and proves the accessor coexists with active media). Bob sends
    // nothing, so the receiver half is dropped rather than drained.
    let audio = handle.audio().await.expect("alice opens the audio stream");
    let (sender, _receiver) = audio.split();
    for index in 0..5u32 {
        let frame = AudioFrame::new(vec![0; 160], 8_000, 1, index * 160);
        if sender.send(frame).await.is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    drop(sender);

    handle.hangup().await.expect("alice hangs up");
    let _ = tokio::time::timeout(Duration::from_secs(5), alice.wait_for_ended(&call_id)).await;

    let bob_media = tokio::time::timeout(Duration::from_secs(10), bob_task)
        .await
        .expect("bob task completes")
        .expect("bob scenario result");
    assert_pcmu(&bob_media, "bob after teardown");

    alice.shutdown().await.expect("alice shuts down");
    bob_shutdown.shutdown();
}