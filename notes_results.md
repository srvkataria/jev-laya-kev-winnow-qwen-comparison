# Result notes

Same 200 synthetic tickets, six queues, cutoff 0.8 fixed in advance. AI Resolution Rate is the share the computer would file. A ticket at or above 0.8 is filed even when the queue is wrong. Accuracy is what shows those mistakes.

A gap of a few points on 200 tickets is noise. A gap of about 10 points is worth saying. A repeated mistake on the same tickets is worth saying even when the headline gap is small.

## Scores

| System | Latency p95 | Accuracy | AI Resolution Rate | Cost per 1,000 | High-confidence mistakes |
|---|---:|---:|---:|---:|---:|
| Qwen | 1,917 ms | 94.0% (188/200) | 100% | $0.0224 | 12 |
| Jev | 367 ms | 100% (200/200) | 97.0% (194/200) | $0.00 | 0 |
| Laya | 50 ms | 70.0% (140/200) | 62.0% | $0.0007 | 9 |
| Laya typed | 95 ms | 75.5% (151/200) | 27.0% | $0.0011 | 0 |
| Kev 0.8B | 170 ms | 84.0% (168/200) | 6.5% | $0.0023 | 0 |
| Kev 4B | 2,009 ms | 92.0% (184/200) | 11.5% | $0.0269 | 0 |
| Winnow e4b | 614 ms | 96.5% (193/200) | 77.0% | $0.0083 | 7 |
| Winnow 12B | 2,341 ms | 96.5% (193/200) | 100% | $0.0240 | 7 |

Jev’s cost cell is $0 because `JEV_COST_PER_CALL_USD` is still 0. The API is not free. Local cost is model time at $0.05 per hour.

## Jev is the only perfect run, and it still holds tickets back

Jev got every ticket right. It auto-filed 194 and left 6 for a person. Those 6 are all refunds, with probabilities from 0.57 to 0.78. One more refund, T-0074, sat exactly on 0.80 and was filed correctly.

No other system matched 200/200. The next best, both Winnows, missed 7.

## Both Winnows miss the same seven refunds, and would file them

Winnow e4b and Winnow 12B got the same 193 tickets right and the same 7 wrong: T-0014, T-0044, T-0074, T-0104, T-0134, T-0164, T-0194. Every one is a refund sent to billing. Example: “Cancel the Team plan from May and return the $49 you took.”

Every one of those seven was above 0.8, so both models would auto-file the mistake. e4b’s probabilities were 0.81–0.85. 12B’s were 0.88–0.96. The larger model did not fix the error. It became more sure of it, then auto-filed the other 193 correct tickets as well, which is why its resolution rate is 100% and e4b’s is 77%.

12B is also slower (p95 2.3 s against 0.6 s) and about three times the local cost. On this set, e4b is the better Winnow.

Jev got all seven refunds right, and kept six of them under 0.8. Qwen, both Layas, and Kev 0.8B also got those seven right. Kev 4B called them billing too, but at about 0.50, so a person would still see them.

## Qwen files everything, including twelve wrong “other” tickets

Qwen’s resolution rate is 100% because every confidence it wrote was at least 0.8. Its twelve mistakes are all `other` tickets, nine sent to billing and three to bugs, each with confidence 0.95. Example: “Please change the email address on account 1. Login and billing both work.”

Mean confidence on Qwen’s wrong tickets is 0.95, against 0.96 on the ones it got right. The number it writes does not mark the misses. Billing, refunds, API access, bugs, and sales were all perfect. The whole 6-point gap from Jev is the `other` queue (21/33).

## Kev gets more accurate as it grows, and still barely auto-files

Kev 0.8B is 84% with a 6.5% resolution rate. Kev 4B is 92% with 11.5%. Neither auto-filed a mistake. Their probabilities sit near 0.5–0.7, so almost every ticket would go to a person, including the ones they got right.

The errors change with size. Kev 0.8B misses bugs (14 sent to `other`) and sales (15 sent to `other`). Kev 4B is perfect on those two and instead confuses refunds with billing (11) and a few billing tickets with `other` (5).

Kev 4B is the slowest expensive local run after Winnow 12B: p95 about 2.0 s, $0.027 per thousand tickets, and the computer would still only take 23 of 200.

## Laya is the fast one, and the typed checkpoint does not catch Jev

Laya answers in about 50 ms and costs the least. Accuracy is 70%. It is perfect on billing and bugs, 91% on refunds, then falls apart on sales (42%) and `other` (6%). Fifty of its sixty misses are those two queues. Nine mistakes were still above 0.8, mostly API-access tickets called bugs.

Laya typed, the fine-tuned checkpoint, moves accuracy to 75.5% and lifts `other` from 6% to 64%. Refunds fall from 91% to 70%, and the resolution rate falls from 62% to 27%. It made no high-confidence mistake. On these tickets it is a bit more careful and a bit more right, not a replacement for Jev or Winnow. The published 0.766 for that checkpoint was on a different benchmark.

## What is worth saying

Jev is the only system that is both fully correct and willing to file most of the pile. It still holds back the refund tickets it is less sure about.

Winnow e4b is the closest local system: 96.5%, 77% filed, 614 ms. The remaining gap is one repeated refund-versus-billing mistake, and e4b would file those seven errors.

Winnow 12B ties e4b on accuracy and is worse on speed, cost, and confidence. A larger model on this set did not mean a better routing system.

Qwen’s 94% is close on accuracy. Its 100% resolution rate is the stated confidence, and that confidence stays at 0.95 on the twelve tickets it gets wrong.

Kev 4B’s 92% comes with an 11.5% resolution rate. The computer declines almost everything. That avoids high-confidence mistakes and also leaves the work with a person.

Laya shows the speed end of the range, about 40 times faster than Winnow 12B at p95, at 70% accuracy.
