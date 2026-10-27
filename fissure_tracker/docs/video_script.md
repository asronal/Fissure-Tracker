# Demo video script (target 4:00, hard limit 5:00)

**Before recording:**
- Run `python tools/make_demo.py`, then `python serve_local.py`, or use the AWS link once deployed.
- Have one printed marker taped next to a real crack, or a crack drawn on paper.
- Record the phone screen and the laptop browser.
- Crop the AWS account number out of any console shots.

| Time | Shot | Say (plain, short) |
|---|---|---|
| 0:00–0:20 | Close-up of a real wall crack, then a hand sticking the printed marker next to it | "Most wall cracks are harmless, but one that keeps growing can mean the building is moving. The advice is to measure it every few weeks. Nobody does." |
| 0:20–0:40 | The marker sheet; the 50 mm check bar with a ruler | "We print a 60 mm marker. It gives every photo a scale in millimetres, the angle, and which crack it is. Marker 1 is crack 1." |
| 0:40–1:10 | Phone: open the page, take a photo, "Measuring the crack…", result page | "Once a week the owner takes a photo in the browser. OpenCV 5 finds the marker, straightens the wall, and measures the crack: maximum width, length, and the widest point." |
| 1:10–1:40 | Scroll the case page: marked-up photo, numbers, the "What the agent did" list | "An AI agent on Amazon Bedrock calls these OpenCV functions as tools, one by one, and every step is shown, with who decided it." |
| 1:40–2:10 | Send a deliberately blurry photo → retake message with the fix | "If the photo isn't good enough, the agent doesn't guess. Blurry? It says exactly how to fix it." |
| 2:10–2:50 | Laptop: crack #1 case (growth → close-up → engineer). Point at steps 3, 4 and 7, 8 | "Here the crack grew from 1.5 to 2.1 millimetres, but the photo was too coarse to be sure. So instead of raising an alarm, the agent asked for a close-up of the widest point, which it located itself. The close-up confirmed it, and only then did it ask a structural engineer." |
| 2:50–3:15 | The engineer e-mail → review page → press "Inspection needed" → owner's page updates | "The engineer gets one link with the photos, numbers and history, and decides. The owner is never told what to do by the AI alone." |
| 3:15–3:35 | "All cracks" page: chart with error bars and the alert line; crack #2 flat | "Over time each crack gets a measured history with its uncertainty. Crack 2 hasn't moved in five weeks." |
| 3:35–3:55 | Architecture diagram | "It runs on AWS Lambda with OpenCV 5 in a layer, Bedrock for the agent, DynamoDB and S3 for history and photos, and SNS for the engineer. It scales to zero and costs a cent or two per photo." |
| 3:55–4:15 | Evaluation tables (docs/evaluation.md) | "On test photos with known answers, printed lines are measured within 0.1 millimetre, and readings repeat within 0.02. Safety rules in code stop the AI from saving a crack that needs an engineer, and a rules backup takes over if the model is down." |
| 4:15–4:25 | Title card: team names | "Thanks for watching." |

**Tips:**
- Record in daylight and keep each screen on for about 2 seconds.
- Cut the waiting time while the photo is measured.
- Keep the total under 5:00.
