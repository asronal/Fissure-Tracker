# Devpost submission draft

> Submitted 27 Oct 2026: https://devpost.com/software/fissure_tracker · Video: https://youtu.be/O0pobe9cLH8 ·
> Track: **Agentic Vision Award**.

**Name:** Fissure Tracker

**Tagline (≤ 200 characters):** A phone photo and a printed marker turn a wall crack into millimetre readings;
an OpenCV 5 agent on AWS decides when to retake, zoom in, or call an engineer.

## Inspiration
Most wall cracks are harmless, but a crack that keeps widening can mean the building is moving. The standard
advice is to measure it every few weeks. Almost nobody does: gauges are fiddly, and a phone photo has no scale.

## What it does
- **Measure:** The owner sticks a printed 60 mm ArUco marker next to the crack and photographs it weekly in a phone browser. OpenCV 5 measures the crack's maximum and typical width, length and widest point in millimetres, with a stated uncertainty.
- **Line up:** It lines the photo up with last week's.
- **Decide:** An agent on Amazon Bedrock (Nova 2 Lite) calls these OpenCV functions as tools and decides the next step:
  - retake with a specific fix
  - close-up of the widest point it located
  - save as stable
  - ask a structural engineer, who approves through a one-time link before the owner is told to act
- **Show:** Every step and who decided it is visible on the case page.

## How we built it
- **OpenCV 5:**
  - ArUco detection, homography-based tilt and a straight-on warp in exact millimetres
  - black-hat crack detection with a noise-based threshold
  - stain rejection
  - thinning plus a Dijkstra main path for length
  - local area / length for width
  - photo quality judged on the marker's edges
  - alignment in millimetres to detect a moved marker or a different viewpoint: ORB first, and OpenCV 5's
    new ALIKED + LightGlue networks when ORB finds too few matches (12 of 13 hard real photo pairs lined up,
    against 9 with ORB alone)
- **Agent:**
  - Bedrock Converse tool use with safety rules enforced in code
  - a rules backup with the same tools when the model is unavailable
  - a full trace of every decision
- **AWS:**
  - Lambda (Python 3.13) with OpenCV in a layer behind a Function URL
  - DynamoDB, a private S3 bucket and SNS
  - least-privilege IAM, a $20 budget alert, and one-command deploy and delete

## Challenges
- OpenCV 5 removed `estimatePoseSingleMarkers`, so we compute tilt from the homography ourselves.
- The pip build has no `ximgproc`, so thinning is our own code.
- Thin cracks can't be measured from far away: instead of guessing, the agent asks for a close-up.
- The rounded edges of dark stains look like cracks: we blank out wide dark areas first.
- ALIKED + LightGlue at first "lined up" photos of a different wall: the network was recognising the
  marker's surroundings. Greying out everything but the wall fixed it.
- The ALIKED model takes a fixed 640 × 640 input, so on its own it was less precise than ORB. We measured
  both on our real photos and made it ORB's backup instead of its replacement.
- The two models (52 MB) don't fit in the Lambda layer next to OpenCV: they live in S3 and are copied to
  `/tmp` when the function starts.
- A close-up can't be lined up with a normal photo: the agent compares with the latest photo of similar scale.

## Accomplishments
- On real iPhone photos of our printed test sheet, lines from 0.3 to 3 mm read within 0.11 mm (mean error
  0.05 mm). On synthetic photos, readings repeat within 0.02 mm.
- The same photo pipeline leads to six different agent paths, depending on what OpenCV sees.

## What we learned
- A number with a known right answer is worth more than a confident label. Test photos with known crack widths let
  us check every change against the right answer instead of eyeballing it.
- Saying "I can't tell" is a result. Most of the agent's value comes from refusing to measure a bad photo and
  asking for exactly the photo it needs.
- Safety rules belong in code, not in the prompt. The model chooses the next tool, but code stops it from closing
  a case that needs an engineer.
- Plan for the model being unavailable. A backup that uses the same tools and the same trace keeps the app
  working through network errors, timeouts and rate limits.
- Small costs add up on serverless: putting OpenCV in a Lambda layer and photos in S3 kept each check to a cent
  or two.

## What's next
- ALIKED + LightGlue at full resolution, so it can go before ORB
- lens calibration
- a Raspberry Pi camera station per crack
- a learned crack segmenter as an extra tool

## Built with
opencv, python, numpy, aws-lambda, amazon-bedrock, amazon-nova, amazon-dynamodb, amazon-s3, amazon-sns, html

## Links
- Try it: `https://hx64dq2waacw4s76cojqxw3d3q0kjien.lambda-url.us-east-1.on.aws/` (access code given to the judges)
- Code: https://github.com/asronal/Fissure-Tracker
- Video: https://youtu.be/O0pobe9cLH8
