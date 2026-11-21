External GPU power log for the descriptive EDGE block (prereg v1.3 version 4, sec. 4.11).
File: gpu_power_20261113_235717.csv (started 2026-11-13 23:57:17 local time, UTC+2), written by
  nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader -lms 1000 -f <file>
Columns (no header row; nvidia-smi truncated the header line written before it started):
  1 timestamp (YYYY/MM/DD HH:MM:SS.mmm, local time)
  2 power.draw [W] (board power)
  3 utilization.gpu [%]
  4 memory.used [MiB] (board memory)
  5 temperature.gpu [C]
  6 clocks.sm [MHz]
Sampling: nominal 1 Hz; writes are buffered (file grows in blocks of several seconds). GPU: NVIDIA RTX 4080 SUPER 16 GB, driver 617.42.

2026-11-14T22:52:32: the first logger (gpu_power_20261113_235717.csv) stopped at 2026-11-14 22:49:33 when the tool host restarted; no GPU job ran between 16:47 and the restart. Logging continues in gpu_power_20261114_225232.csv (same command and columns).
2026-11-16T01:48:37: the second logger (gpu_power_20261114_225232.csv) stopped at 2026-11-16 01:41:19 when the tool host restarted; the queue was down from about 01:42 until its relaunch. Logging continues in gpu_power_20261116_014837.csv (same command and columns).
2026-11-16T08:09:47: logger gpu_power_20261116_014837.csv stopped on purpose for the user's planned pause (queue stopped by results\STOP; last job c1_moe_b2_books_d5_A ended, chain6b exit 2 at 08:09:03). No queue job runs during the pause; logging restarts in a new file at the resume (same command and columns).
2026-11-16T13:58:33: logging resumes after the user's planned pause (08:09-14:00) in gpu_power_20261116_135833.csv (same command and columns); the A-2 GPU check (13:53:51-13:54:36) ran before this logger started.
2026-11-17T10:19:08: logger gpu_power_20261116_135833.csv was stopped on purpose at about 2026-11-16 18:56 for the user's second pause (machine shutdown; queue stopped by results\STOP after c1_moe_b2_books_d1_B, chain6b exit 2 at 18:55:25). No queue job ran during the pause. Logging resumes in gpu_power_20261117_101908.csv (same command and columns).
2026-11-17T15:36:33: logger gpu_power_20261117_101908.csv stopped on purpose at about 15:36 for the user's third pause (after the core; queue stopped by results\STOP after confirm_base_moe_b3, chain6b exit 2 at 15:36:02). No queue job runs during the pause; logging restarts in a new file at the resume (same command and columns).2026-11-18T03:15:53: logging resumes after the third pause on the author's command to run all remaining registered blocks, in gpu_power_20261118_031553.csv (same command and columns). Two exploratory post-freeze GPU blocks outside the registered queue (random-direction control extension; Muon step-1 displacement) ran from 2026-11-17T23:23 to 2026-11-18T03:15:53 without this logger.
2026-11-18T12:13:37: logger gpu_power_20261118_031553.csv stopped after the chain ended (exit 2).
2026-11-18T14:34:10: logging resumes after the fourth pause (queue stopped by STOP before position 122 at 2026-11-18T12:13:37; system shut down by the project owner) in gpu_power_20261118_143410.csv (same command and columns).
2026-11-19T12:14:36: logging resumes in gpu_power_20261119_121436.csv (same command and columns) after a restart of the launching software at about 2026-11-19T11:40 ended the chain and the previous logger (gpu_power_20261118_143410.csv, last line 11:39:33); the interrupted job dose_moe_b1_books_d3_A is repeated from its start.
2026-11-19T20:34:42: logging resumes in gpu_power_20261119_203442.csv (same command and columns) after a restart of the launching software at about 2026-11-19T20:26-20:31 ended the chain and the previous logger (gpu_power_20261119_121436.csv); the interrupted job c1de_moe_b1_d2_B is repeated from its start.
2026-11-19T21:07:02: logging resumes in gpu_power_20261119_210702.csv (same command and columns) after a reboot of the computer at about 2026-11-19T21:02 ended the chain and the previous logger (gpu_power_20261119_203442.csv); the interrupted job c1de_moe_b1_d2_B is repeated from its start.
2026-11-20T06:00:09: logger gpu_power_20261119_210702.csv stopped after the chain ended (exit 0).
2026-11-20T06:05:46 (watch_and_rerun.ps1, amendment A-3): logging resumes for the re-run of audit_moe_b1 and audit_moe_b2 after the main queue in gpu_power_20261120_060541.csv (same command and columns; logger PID 36960).
2026-11-20T06:39:01 (results\incident_i7\resume_i7.ps1, incident I-7): logging resumes for the repetition of the A-3 re-run of audit_moe_b1 and audit_moe_b2 after a restart of the computer in gpu_power_20261120_063856.csv (same command and columns; logger PID 11888).
2026-11-20T13:05:01 (watch_and_rerun_a4.ps1, amendment A-4): logging resumes for the re-run of c3_freeze_moe_b1_german_s103 and c3_freeze_nestor_german_s103 after the A-3 audit re-run in gpu_power_20261120_130455.csv (same command and columns; logger PID 13636).
