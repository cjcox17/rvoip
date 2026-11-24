//! One continuous RTP timeline per outgoing stream.
//!
//! A session sends on one SSRC with one sequence space, but the payloads it
//! sends can come from several producers with unrelated RTP clocks: the
//! application's own audio (prompts, music), a peer whose packets are relayed
//! or transcoded through a bridge, a later peer after the bridge is replaced,
//! DTMF and comfort noise. Forwarding each producer's timestamps unchanged
//! makes the stream jump by the difference between two unrelated clocks at
//! every switch, backwards about half the time, and receivers' jitter
//! buffers then discard the audio that follows as late.
//!
//! [`OutgoingTimeline`] rebases every producer onto the stream's own
//! timeline. Each producer is identified by a [`TimelineSource`]. While one
//! media source stays current its timestamps keep a fixed offset, so its
//! relative timing (packet spacing, silence gaps, reordering) is preserved
//! exactly. On a source change, or a discontinuity in the current source's
//! clock, the source is re-anchored just after the last timestamp sent,
//! advanced by the wall time that passed, and the packet is marked as the
//! start of a talkspurt (RFC 3550 §5.1). A source whose timestamps already
//! sit on the stream's timeline (for example one stamped from
//! [`OutgoingTimeline::next_timestamp`]) is kept as is.
//!
//! The first packet a stream ever sends keeps its timestamp, so a stream with
//! a single producer is sent exactly as stamped.
//!
//! RFC 4733 events are placed, not rebased per packet: every packet of one
//! event shares the start timestamp the event was given, and a new event
//! starts at the current media position.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::RtpTimestamp;

/// Half the RTP timestamp space: the wrap-aware "is ahead" boundary.
///
/// A relayed source's step back of up to 300 ms (or three packets, if
/// longer) is a late packet; a longer one is a clock reset.
const HALF: u32 = 1 << 31;

/// Which producer stamped a packet sent on a session.
///
/// Packets sent through the plain send APIs come from
/// [`TimelineSource::LOCAL`]. A producer that relays another clock onto the
/// session (such as a bridge) takes its own origin from
/// [`TimelineSource::new_origin`], and a separate stream within it (the
/// relayed peer's SSRC) with [`TimelineSource::with_stream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimelineSource {
    origin: u64,
    stream: u32,
}

impl TimelineSource {
    /// The session's own producer: the application's audio, DTMF and
    /// comfort noise sent through the plain send APIs.
    pub const LOCAL: Self = Self {
        origin: 0,
        stream: 0,
    };

    /// A process-unique producer, distinct from [`Self::LOCAL`] and every
    /// other origin.
    pub fn new_origin() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            origin: NEXT.fetch_add(1, Ordering::Relaxed),
            stream: 0,
        }
    }

    /// The same producer, for one of the streams it relays.
    pub fn with_stream(self, stream: u32) -> Self {
        Self { stream, ..self }
    }
}

/// How a payload's timestamp relates to the stream clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PacketClass {
    /// Audio and comfort noise: the timestamp advances with sampling.
    Media,
    /// RFC 4733 events and any other payload: every packet of one event
    /// carries the event's start timestamp.
    Event,
}

#[derive(Debug, Clone, Copy)]
struct MediaState {
    source: TimelineSource,
    /// Output = source timestamp + offset (wrapping).
    offset: u32,
    last_source: u32,
    last_output: u32,
    sent_at: Instant,
    /// The source's packet duration, learned from its timestamp steps.
    frame_ticks: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct EventState {
    source: TimelineSource,
    source_timestamp: u32,
    output: u32,
}

/// Rebases the timestamps of every packet sent on one SSRC onto one
/// continuous timeline. See the module documentation.
#[derive(Debug, Default)]
pub(crate) struct OutgoingTimeline {
    clock_rate: u32,
    media: Option<MediaState>,
    event: Option<EventState>,
    /// The newest timestamp sent, of any class, and when.
    newest: Option<(u32, Instant)>,
}

/// A mapped packet: the timestamp to send and whether to set the marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mapped {
    pub timestamp: RtpTimestamp,
    pub marker: bool,
}

/// `a` is at or after `b` on a wrapping RTP clock.
fn at_or_after(a: u32, b: u32) -> bool {
    a.wrapping_sub(b) < HALF
}

fn later(a: u32, b: u32) -> u32 {
    if at_or_after(a, b) {
        a
    } else {
        b
    }
}

fn ticks(elapsed: Duration, clock_rate: u32) -> u32 {
    (elapsed.as_nanos() * u128::from(clock_rate) / 1_000_000_000).min(u128::from(HALF / 2)) as u32
}

impl OutgoingTimeline {
    /// Map one packet onto the stream timeline and record it as sent.
    pub(crate) fn map(
        &mut self,
        source: TimelineSource,
        timestamp: RtpTimestamp,
        class: PacketClass,
        clock_rate: u32,
        now: Instant,
    ) -> Mapped {
        let clock_rate = clock_rate.max(1);
        if clock_rate != self.clock_rate {
            // A renegotiated clock makes every learned offset and duration
            // meaningless; the next media packet re-anchors.
            self.clock_rate = clock_rate;
            if let Some(media) = &mut self.media {
                media.source = TimelineSource::new_origin();
                media.frame_ticks = None;
            }
        }
        let mapped = match class {
            PacketClass::Media => self.map_media(source, timestamp, now),
            PacketClass::Event => self.map_event(source, timestamp, now),
        };
        self.newest = Some(match self.newest {
            Some((newest, at)) if !at_or_after(mapped.timestamp, newest) => (newest, at),
            _ => (mapped.timestamp, now),
        });
        mapped
    }

    /// The timestamp a media packet sent at `now` from a new producer would
    /// be given: just after the last packet sent, advanced by the wall time
    /// since. `None` until the stream has sent anything.
    ///
    /// A producer that stamps its own audio from this value joins the
    /// stream's timeline as is, so its timestamps match the wire.
    pub(crate) fn next_timestamp(&self, clock_rate: u32, now: Instant) -> Option<RtpTimestamp> {
        let clock_rate = clock_rate.max(1);
        let (base, base_at, frame) = self.base(clock_rate)?;
        let step = frame.max(ticks(now.saturating_duration_since(base_at), clock_rate));
        Some(self.not_before_newest(base.wrapping_add(step)))
    }

    fn default_frame(&self) -> u32 {
        // 20 ms, the usual packet duration.
        (self.clock_rate / 50).max(1)
    }

    /// Where the timeline last stood: the last media packet, or the newest
    /// packet of any class if no media was sent yet.
    fn base(&self, clock_rate: u32) -> Option<(u32, Instant, u32)> {
        let default_frame = (clock_rate / 50).max(1);
        if let Some(media) = &self.media {
            return Some((
                media.last_output,
                media.sent_at,
                media.frame_ticks.unwrap_or(default_frame),
            ));
        }
        self.newest.map(|(ts, at)| (ts, at, default_frame))
    }

    fn not_before_newest(&self, timestamp: u32) -> u32 {
        match self.newest {
            Some((newest, _)) => later(timestamp, newest),
            None => timestamp,
        }
    }

    fn map_media(&mut self, source: TimelineSource, timestamp: u32, now: Instant) -> Mapped {
        let current = self.media.filter(|media| media.source == source);
        let Some(mut media) = current else {
            return self.reanchor(source, timestamp, now, None);
        };
        let frame = media.frame_ticks.unwrap_or(self.default_frame());
        let elapsed = ticks(
            now.saturating_duration_since(media.sent_at),
            self.clock_rate,
        );
        let delta = timestamp.wrapping_sub(media.last_source);
        if delta >= HALF {
            let back = delta.wrapping_neg();
            // A relayed source's packets can arrive out of order; the
            // session's own producer sends in order, so any step back from
            // it is a new clock.
            let late_window = if source == TimelineSource::LOCAL {
                0
            } else {
                frame
                    .saturating_mul(3)
                    .max(self.clock_rate.saturating_mul(3) / 10)
            };
            if back <= late_window {
                // A packet the source sent earlier, arriving late. Keep its
                // relative timing, and the stream's state, as they were.
                return Mapped {
                    timestamp: timestamp.wrapping_add(media.offset),
                    marker: false,
                };
            }
            // The source's clock stepped back: a reset, not reordering.
            return self.reanchor(source, timestamp, now, media.frame_ticks);
        }
        if delta > elapsed.saturating_add(self.clock_rate) {
            // The clock stepped forward far beyond the time that passed.
            return self.reanchor(source, timestamp, now, media.frame_ticks);
        }
        let talkspurt = delta > frame.saturating_mul(2);
        if !talkspurt && delta > 0 && delta <= self.clock_rate / 8 {
            // The shortest step seen is the packet duration; longer steps
            // are loss or silence.
            media.frame_ticks = Some(media.frame_ticks.map_or(delta, |frame| frame.min(delta)));
        }
        let output = timestamp.wrapping_add(media.offset);
        media.last_source = timestamp;
        media.last_output = output;
        media.sent_at = now;
        self.media = Some(media);
        Mapped {
            timestamp: output,
            marker: talkspurt,
        }
    }

    /// Start `source` on the stream timeline with `timestamp`.
    fn reanchor(
        &mut self,
        source: TimelineSource,
        timestamp: u32,
        now: Instant,
        frame_ticks: Option<u32>,
    ) -> Mapped {
        let output = match self.base(self.clock_rate) {
            // The stream's first packet keeps its timestamp.
            None => timestamp,
            Some((base, base_at, frame)) => {
                let step = frame.max(ticks(
                    now.saturating_duration_since(base_at),
                    self.clock_rate,
                ));
                let ahead = timestamp.wrapping_sub(base);
                let on_timeline = ahead > 0
                    && ahead < HALF
                    && ahead <= step.saturating_add(self.clock_rate / 10)
                    && self
                        .newest
                        .is_none_or(|(newest, _)| at_or_after(timestamp, newest));
                if on_timeline {
                    timestamp
                } else {
                    self.not_before_newest(base.wrapping_add(step))
                }
            }
        };
        self.media = Some(MediaState {
            source,
            offset: output.wrapping_sub(timestamp),
            last_source: timestamp,
            last_output: output,
            sent_at: now,
            frame_ticks,
        });
        Mapped {
            timestamp: output,
            marker: true,
        }
    }

    fn map_event(&mut self, source: TimelineSource, timestamp: u32, now: Instant) -> Mapped {
        if let Some(event) = self.event {
            if event.source == source && event.source_timestamp == timestamp {
                // Every packet of one event shares its start timestamp.
                return Mapped {
                    timestamp: event.output,
                    marker: false,
                };
            }
        }
        let output = match self.base(self.clock_rate) {
            None => timestamp,
            Some((base, base_at, frame)) => {
                let elapsed = ticks(now.saturating_duration_since(base_at), self.clock_rate);
                // An event from the current media source keeps its place
                // on that source's clock when that place is plausible.
                let candidate = self
                    .media
                    .filter(|media| media.source == source)
                    .map(|media| timestamp.wrapping_add(media.offset))
                    .filter(|candidate| {
                        let ahead = candidate.wrapping_sub(base);
                        ahead < HALF
                            && ahead <= elapsed.saturating_add(frame.saturating_mul(2))
                            && self
                                .newest
                                .is_none_or(|(newest, _)| at_or_after(*candidate, newest))
                    });
                candidate.unwrap_or_else(|| {
                    // Otherwise the event starts where the next media packet
                    // will: one frame on while media flows, or the wall time
                    // since the last media packet once it has stopped.
                    let step = if elapsed < frame.saturating_mul(2) {
                        frame
                    } else {
                        elapsed
                    };
                    self.not_before_newest(base.wrapping_add(step))
                })
            }
        };
        self.event = Some(EventState {
            source,
            source_timestamp: timestamp,
            output,
        });
        Mapped {
            timestamp: output,
            marker: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PCMU: u32 = 8_000;
    const OPUS: u32 = 48_000;

    struct Harness {
        timeline: OutgoingTimeline,
        start: Instant,
        rate: u32,
        sent: Vec<(u32, bool)>,
    }

    impl Harness {
        fn new(rate: u32) -> Self {
            Self {
                timeline: OutgoingTimeline::default(),
                start: Instant::now(),
                rate,
                sent: Vec::new(),
            }
        }

        fn send(&mut self, at_ms: u64, source: TimelineSource, ts: u32, class: PacketClass) -> u32 {
            let mapped = self.timeline.map(
                source,
                ts,
                class,
                self.rate,
                self.start + Duration::from_millis(at_ms),
            );
            self.sent.push((mapped.timestamp, mapped.marker));
            mapped.timestamp
        }

        fn media(&mut self, at_ms: u64, source: TimelineSource, ts: u32) -> u32 {
            self.send(at_ms, source, ts, PacketClass::Media)
        }

        /// `count` 20 ms packets from `source` starting at `ts`, one every
        /// 20 ms of wall time from `at_ms`.
        fn run(&mut self, at_ms: u64, source: TimelineSource, ts: u32, count: u32) -> Vec<u32> {
            let frame = self.rate / 50;
            (0..count)
                .map(|i| {
                    self.media(
                        at_ms + u64::from(i) * 20,
                        source,
                        ts.wrapping_add(i * frame),
                    )
                })
                .collect()
        }

        fn assert_never_backwards(&self) {
            for pair in self.sent.windows(2) {
                assert!(
                    at_or_after(pair[1].0, pair[0].0),
                    "timestamp went back from {} to {} in {:?}",
                    pair[0].0,
                    pair[1].0,
                    self.sent
                );
            }
        }
    }

    fn steps(outputs: &[u32]) -> Vec<u32> {
        outputs
            .windows(2)
            .map(|pair| pair[1].wrapping_sub(pair[0]))
            .collect()
    }

    #[test]
    fn a_single_source_is_sent_exactly_as_stamped() {
        let mut h = Harness::new(PCMU);
        let out = h.run(0, TimelineSource::LOCAL, 123_456, 10);
        assert_eq!(out[0], 123_456);
        assert!(steps(&out).iter().all(|step| *step == 160));
        // Only the first packet starts a talkspurt.
        assert!(h.sent[0].1);
        assert!(h.sent[1..].iter().all(|(_, marker)| !marker));
    }

    #[test]
    fn switching_sources_continues_forward_with_a_marker() {
        for rate in [PCMU, OPUS] {
            for far_clock in [0, 1_000, HALF - 5, HALF + 5, u32::MAX - 100] {
                let mut h = Harness::new(rate);
                let frame = rate / 50;
                let bridge = TimelineSource::new_origin().with_stream(0xAAAA);
                let local = h.run(0, TimelineSource::LOCAL, 50_000, 5);
                let relayed = h.run(100, bridge, far_clock, 5);
                let again = h.run(200, TimelineSource::LOCAL, 7, 5);
                h.assert_never_backwards();
                assert_eq!(relayed[0], local[4].wrapping_add(frame));
                assert_eq!(again[0], relayed[4].wrapping_add(frame));
                // Each source's own spacing is kept, and each switch is
                // marked.
                for run in [&local, &relayed, &again] {
                    assert!(steps(run).iter().all(|step| *step == frame));
                }
                assert!(h.sent[5].1 && h.sent[10].1);
            }
        }
    }

    #[test]
    fn the_wall_time_between_sources_is_kept() {
        let mut h = Harness::new(PCMU);
        let bridge = TimelineSource::new_origin().with_stream(1);
        let first = h.run(0, bridge, 9_000, 3);
        // Nothing for one second, then a prompt.
        let prompt = h.run(1_040, TimelineSource::LOCAL, 0, 3);
        assert_eq!(prompt[0].wrapping_sub(first[2]), 8_000);
        h.assert_never_backwards();
    }

    #[test]
    fn relayed_silence_loss_and_reordering_keep_their_relative_timing() {
        let mut h = Harness::new(PCMU);
        let src = TimelineSource::new_origin().with_stream(9);
        let base = h.media(0, TimelineSource::LOCAL, 1_000);
        let a = h.media(20, src, 40_000);
        assert_eq!(a, base + 160);
        // 2 s of discontinuous transmission, then the talkspurt resumes.
        let b = h.media(2_020, src, 40_000 + 16_000);
        assert_eq!(b - a, 16_000);
        assert!(h.sent.last().unwrap().1, "a talkspurt starts with a marker");
        // A lost packet keeps its gap.
        let c = h.media(2_060, src, 40_000 + 16_000 + 320);
        assert_eq!(c - b, 320);
        // A late packet keeps its place and does not move the source.
        let late = h.media(2_065, src, 40_000 + 16_000 + 160);
        assert_eq!(late, b + 160);
        let d = h.media(2_080, src, 40_000 + 16_000 + 480);
        assert_eq!(d - c, 160);
    }

    #[test]
    fn a_deeply_late_relayed_packet_keeps_its_place_without_moving_the_source() {
        let mut h = Harness::new(PCMU);
        let src = TimelineSource::new_origin().with_stream(4);
        let out = h.run(0, src, 70_000, 12);
        // A packet from 200 ms (ten frames) back arrives now.
        let late = h.media(240, src, 70_000 + 160);
        assert_eq!(late, out[1]);
        assert!(!h.sent.last().unwrap().1);
        // The source goes on with its own spacing and no new marker.
        let next = h.media(240, src, 70_000 + 12 * 160);
        assert_eq!(next, out[11] + 160);
        assert!(!h.sent.last().unwrap().1);
    }

    #[test]
    fn a_step_back_from_the_local_producer_is_a_new_clock() {
        let mut h = Harness::new(PCMU);
        let out = h.run(0, TimelineSource::LOCAL, 90_000, 3);
        // A prompt restarting its clock 40 ms back is a new clock, not a
        // late packet.
        let again = h.media(60, TimelineSource::LOCAL, 90_000);
        assert_eq!(again, out[2] + 160);
        assert!(h.sent.last().unwrap().1);
        h.assert_never_backwards();
    }

    #[test]
    fn a_source_clock_reset_or_jump_reanchors() {
        let mut h = Harness::new(PCMU);
        let out = h.run(0, TimelineSource::LOCAL, 500_000, 3);
        // Back to zero (a prompt player restarting its clock).
        let reset = h.media(60, TimelineSource::LOCAL, 0);
        assert_eq!(reset, out[2] + 160);
        // A forward jump of an hour within 20 ms.
        let jump = h.media(80, TimelineSource::LOCAL, 8_000 * 3_600);
        assert_eq!(jump, reset + 160);
        h.assert_never_backwards();
    }

    #[test]
    fn timestamps_wrap() {
        let mut h = Harness::new(PCMU);
        let out = h.run(0, TimelineSource::LOCAL, u32::MAX - 200, 4);
        assert_eq!(out[3], (u32::MAX - 200).wrapping_add(480));
        let bridge = TimelineSource::new_origin();
        let relayed = h.run(80, bridge, 77, 3);
        assert_eq!(relayed[0], out[3].wrapping_add(160));
        h.assert_never_backwards();
    }

    #[test]
    fn a_producer_stamped_from_next_timestamp_is_kept_as_is() {
        let mut h = Harness::new(OPUS);
        let bridge = TimelineSource::new_origin().with_stream(3);
        h.run(0, bridge, 2_000_000_000, 5);
        let now = h.start + Duration::from_millis(300);
        let next = h.timeline.next_timestamp(OPUS, now).unwrap();
        let prompt = h.run(300, TimelineSource::LOCAL, next, 5);
        assert_eq!(prompt[0], next);
        assert_eq!(prompt[4], next + 4 * 960);
        h.assert_never_backwards();
    }

    #[test]
    fn next_timestamp_is_none_before_anything_is_sent() {
        assert_eq!(
            OutgoingTimeline::default().next_timestamp(PCMU, Instant::now()),
            None
        );
    }

    #[test]
    fn a_relayed_event_keeps_one_timestamp_and_its_place() {
        let mut h = Harness::new(PCMU);
        let src = TimelineSource::new_origin().with_stream(5);
        let audio = h.run(0, src, 10_000, 3);
        // The far side starts an event at its next audio timestamp and
        // repeats that start timestamp in every packet of the event.
        let event: Vec<u32> = (0..6)
            .map(|i| h.send(60 + i * 20, src, 10_480, PacketClass::Event))
            .collect();
        assert!(event.iter().all(|ts| *ts == audio[2] + 160));
        let after = h.media(180, src, 10_480 + 960);
        assert_eq!(after, audio[2] + 160 + 960);
        h.assert_never_backwards();
    }

    #[test]
    fn a_local_event_during_relayed_audio_starts_at_the_next_media_slot() {
        let mut h = Harness::new(PCMU);
        let src = TimelineSource::new_origin().with_stream(5);
        let audio = h.run(0, src, 10_000, 3);
        // A local transmitter stamps the event from an unrelated clock.
        let start = h.send(45, TimelineSource::LOCAL, 3_000_000_000, PacketClass::Event);
        assert_eq!(start, audio[2] + 160);
        let repeat = h.send(65, TimelineSource::LOCAL, 3_000_000_000, PacketClass::Event);
        assert_eq!(repeat, start);
        // Relayed audio goes on around it, never before the event start.
        let more = h.run(60, src, 10_480, 3);
        assert_eq!(more[0], start);
        h.assert_never_backwards();
    }

    #[test]
    fn an_event_after_a_pause_starts_at_the_wall_time() {
        let mut h = Harness::new(PCMU);
        let audio = h.run(0, TimelineSource::LOCAL, 0, 2);
        let start = h.send(1_020, TimelineSource::LOCAL, 1, PacketClass::Event);
        assert_eq!(start, audio[1] + 8_000);
    }

    #[test]
    fn a_renegotiated_clock_reanchors_forward() {
        let mut h = Harness::new(PCMU);
        let out = h.run(0, TimelineSource::LOCAL, 1_000, 3);
        h.rate = OPUS;
        let after = h.run(60, TimelineSource::LOCAL, 1_000 + 480, 3);
        assert!(at_or_after(after[0], out[2].wrapping_add(1)));
        assert!(steps(&after).iter().all(|step| *step == 960));
        assert!(h.sent[3].1);
    }
}
