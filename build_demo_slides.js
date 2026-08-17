/*
 * build_demo_slides.js  -  8-slide deck for the midterm (preliminary) demo video.
 * Scoped to the midterm prototype only. No live-demo slide (talked over instead).
 * Slower, deliberate delivery: roughly 35-40 seconds per slide, about 5 minutes.
 * Plain language, one idea per slide, consistent UI, no em dashes.
 * Run:  NODE_PATH=<pptxgenjs> node build_demo_slides.js
 */
const pptxgen = require("pptxgenjs");
const path = require("path");
const FIG = path.join(__dirname, "outputs", "figures");

const NAVY = "1E2761", ICE = "CADCFC", WHITE = "FFFFFF",
      GOLD = "F2B705", INK = "1B2333", SLATE = "5B6472", LIGHT = "F7F9FC";
const TOTAL = 8;

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "Scott Chen";
p.title = "Profit-shifting risk: prototype demo";
const W = 13.33, H = 7.5;

// ---- shared scaffolding so every slide feels part of one deck ---- //
function footer(s, n) {
  s.addText("Detecting profit-shifting risk", { x: 0.7, y: 7.02, w: 7, h: 0.3,
    fontFace: "Calibri", fontSize: 9.5, color: SLATE, margin: 0 });
  s.addText(`${n} / ${TOTAL}`, { x: 11.8, y: 7.02, w: 0.83, h: 0.3,
    fontFace: "Calibri", fontSize: 9.5, color: SLATE, align: "right", margin: 0 });
}
function content(n, kicker, title) {
  const s = p.addSlide();
  s.background = { color: LIGHT };
  // recurring motif: small gold square + section label, top-left
  s.addShape(p.shapes.RECTANGLE, { x: 0.7, y: 0.62, w: 0.16, h: 0.16, fill: { color: GOLD } });
  s.addText(kicker.toUpperCase(), { x: 0.98, y: 0.55, w: 11, h: 0.32,
    fontFace: "Calibri", fontSize: 12.5, color: SLATE, bold: true, charSpacing: 2, margin: 0 });
  s.addText(title, { x: 0.68, y: 0.95, w: 12.0, h: 0.95, fontFace: "Georgia",
    fontSize: 30, color: NAVY, bold: true, margin: 0 });
  footer(s, n);
  return s;
}
function dark() {
  const s = p.addSlide();
  s.background = { color: NAVY };
  return s;
}
const note = (s, t) => s.addNotes(t);
const mkShadow = () => ({ type: "outer", color: "888888", blur: 6, offset: 2, angle: 135, opacity: 0.18 });

// ===================== 1. TITLE (dark) ===================== //
let s = dark();
s.addShape(p.shapes.RECTANGLE, { x: 0.9, y: 2.35, w: 0.55, h: 0.16, fill: { color: GOLD } });
s.addText("Spotting where companies may be hiding profit", { x: 0.9, y: 2.6, w: 11.6, h: 1.7,
  fontFace: "Georgia", fontSize: 40, color: WHITE, bold: true, margin: 0 });
s.addText("A machine-learning prototype built on public OECD company data", {
  x: 0.9, y: 4.35, w: 11.4, h: 0.6, fontFace: "Calibri", fontSize: 19, color: ICE, margin: 0 });
s.addText([
  { text: "Scott Chen", options: { bold: true, color: WHITE } },
  { text: "    CM3070 Final Project, preliminary submission", options: { color: ICE } },
], { x: 0.9, y: 6.2, w: 11.5, h: 0.4, fontFace: "Calibri", fontSize: 14, margin: 0 });
note(s, "Hi, I am Scott. This is the demo of the working prototype for my final project, "
  + "submitted at the halfway point. The project is about spotting where big companies may "
  + "be hiding profit to pay less tax, using only public data. I will keep this to the "
  + "prototype and what it shows so far.");

// ===================== 2. EXECUTIVE SUMMARY (dark) ===================== //
s = dark();
s.addShape(p.shapes.RECTANGLE, { x: 0.9, y: 0.7, w: 0.16, h: 0.16, fill: { color: GOLD } });
s.addText("IN ONE MINUTE", { x: 1.18, y: 0.63, w: 8, h: 0.32, fontFace: "Calibri",
  fontSize: 12.5, color: GOLD, bold: true, charSpacing: 2, margin: 0 });
s.addText("The whole project at a glance", { x: 0.88, y: 1.0, w: 11.5, h: 0.8,
  fontFace: "Georgia", fontSize: 30, color: WHITE, bold: true, margin: 0 });
const row = (y, head, body) => {
  s.addShape(p.shapes.RECTANGLE, { x: 0.95, y: y + 0.06, w: 0.07, h: 0.9, fill: { color: GOLD } });
  s.addText(head, { x: 1.2, y, w: 3.4, h: 1.0, fontFace: "Calibri", fontSize: 17, bold: true, color: WHITE, margin: 0, valign: "middle" });
  s.addText(body, { x: 4.7, y, w: 7.7, h: 1.0, fontFace: "Calibri", fontSize: 15, color: ICE, margin: 0, valign: "middle" });
};
row(2.15, "The question", "Can a model flag likely tax-avoidance hotspots from public data alone?");
row(3.25, "What I built", "A pipeline that scores every country-pair from 0 to 1 for risk, using four models.");
row(4.35, "The new idea", "Use what kinds of companies sit in a place: empty shells versus real operations.");
row(5.45, "Result so far", "It works: it ranks known tax havens at the top, with an AUC of 0.94.");
note(s, "Here is the whole thing in one minute, so the rest of the talk makes sense. The "
  + "question: can a model flag likely tax-avoidance hotspots from public data alone? What I "
  + "built: a pipeline that gives every country-pair a risk score from zero to one, using "
  + "four different models. The new idea I am testing: use what kinds of companies sit in a "
  + "place, empty holding shells versus real operations. And the result so far is positive, "
  + "which I will show you.");

// ===================== 3. THE PROBLEM (light) ===================== //
s = content(3, "Why it matters", "Profit on paper, not where the work is");
s.addText([
  { text: "A big multinational is many companies in different countries, one owner.", options: { bullet: true, breakLine: true } },
  { text: "Because they trade with each other, the group can choose where profit lands.", options: { bullet: true, breakLine: true } },
  { text: "The incentive: park profit where tax is low, even with no real activity there.", options: { bullet: true, breakLine: true } },
  { text: "Old rules (such as a simple low-tax-rate flag) miss cases that hide in combinations.", options: { bullet: true } },
], { x: 0.7, y: 2.15, w: 7.5, h: 3.6, fontFace: "Calibri", fontSize: 17, color: INK, paraSpaceAfter: 14 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 8.7, y: 2.5, w: 3.9, h: 2.7, fill: { color: NAVY }, rectRadius: 0.12, shadow: mkShadow() });
s.addText("~40%", { x: 8.7, y: 2.78, w: 3.9, h: 1.1, align: "center", fontFace: "Georgia", fontSize: 58, color: GOLD, bold: true, margin: 0 });
s.addText("of multinational profit is shifted to tax havens each year", { x: 8.95, y: 3.95, w: 3.4, h: 0.9, align: "center", fontFace: "Calibri", fontSize: 14, color: ICE, margin: 0 });
s.addText("Source: Torslov, Wier and Zucman (2022)", { x: 8.95, y: 4.8, w: 3.4, h: 0.3, align: "center", fontFace: "Calibri", fontSize: 10, italic: true, color: ICE, margin: 0 });
note(s, "First, why this matters. A big multinational is really many companies in different "
  + "countries under one owner, and because they trade with each other, the group has a lot "
  + "of say over where its profit lands on paper. The incentive is obvious: park profit where "
  + "tax is low, even if no real work happens there. The old screening rules, like flagging a "
  + "low tax rate, miss the tricky cases where no single number looks wrong but the "
  + "combination does. Researchers estimate around 40% of multinational profit is shifted "
  + "this way each year, so the stakes are large.");

// ===================== 4. DATA & NEW IDEA (light) ===================== //
s = content(4, "Data and idea", "Public data, plus a signal others skip");
s.addText([
  { text: "Built only on free OECD data: about 14,000 country-pairs, 2016 to 2021.", options: { bullet: true, breakLine: true } },
  { text: "Most studies stop at the money: profit, tax, revenue.", options: { bullet: true, breakLine: true } },
  { text: "I also use the mix of company types in each place: holding shells, finance, dormant.", options: { bullet: true, breakLine: true } },
  { text: "The label is whether a place is a known tax haven. The model never sees its name.", options: { bullet: true } },
], { x: 0.7, y: 2.15, w: 12, h: 2.6, fontFace: "Calibri", fontSize: 17, color: INK, paraSpaceAfter: 13 });
const chip = (x, big, lab) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 5.05, w: 3.8, h: 1.55, fill: { color: WHITE }, line: { color: ICE, width: 1.25 }, rectRadius: 0.1, shadow: mkShadow() });
  s.addText(big, { x, y: 5.2, w: 3.8, h: 0.7, align: "center", fontFace: "Georgia", fontSize: 25, color: NAVY, bold: true, margin: 0 });
  s.addText(lab, { x: x + 0.2, y: 5.9, w: 3.4, h: 0.6, align: "center", fontFace: "Calibri", fontSize: 12, color: SLATE, margin: 0 });
};
chip(0.7, "0.11 vs 0.28", "tax actually paid: haven vs ordinary");
chip(4.75, "0.21 vs 0.04", "share that are holding shells");
chip(8.8, "38 vs 152", "staff per company");
note(s, "The project runs entirely on free OECD data: about 14,000 country-pairs over six "
  + "years. Most studies stop at the money, profit, tax and revenue. My addition is to also "
  + "look at what kinds of companies sit in each place: empty holding shells, internal "
  + "finance vehicles, dormant ones, versus real operating businesses. The bottom row shows "
  + "how different havens already look: they pay far less tax, hold far more empty shells, "
  + "and run on a fraction of the staff. The model has to learn this without ever being told "
  + "the country's name, so it cannot just memorise the answer.");

// ===================== 5. WHAT I BUILT (light) ===================== //
s = content(5, "The prototype", "One pipeline, four models, one score");
const box = (x, t, d) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 2.7, w: 2.7, h: 2.0, fill: { color: NAVY }, rectRadius: 0.1, shadow: mkShadow() });
  s.addText(t, { x: x + 0.18, y: 2.92, w: 2.35, h: 0.6, fontFace: "Calibri", fontSize: 15, bold: true, color: WHITE, margin: 0 });
  s.addText(d, { x: x + 0.18, y: 3.55, w: 2.35, h: 1.05, fontFace: "Calibri", fontSize: 11.5, color: ICE, margin: 0 });
};
box(0.7, "1. Tidy the data", "925k raw rows into 14k clean ones");
box(3.75, "2. Build signals", "18 features, incl. the company-type mix");
box(6.8, "3. Train models", "tree model, two neural nets, anomaly detector");
box(9.85, "4. Score and explain", "risk 0 to 1, plus why it flagged each case");
for (const ax of [3.5, 6.55, 9.6]) s.addText("→", { x: ax, y: 3.35, w: 0.3, h: 0.6, fontFace: "Arial", fontSize: 26, color: SLATE, bold: true, margin: 0 });
s.addText("It is reproducible and runs end to end on a laptop with one command.", {
  x: 0.7, y: 5.25, w: 12, h: 0.4, fontFace: "Calibri", fontSize: 15.5, italic: true, color: INK, margin: 0 });
note(s, "Here is the prototype on one slide. Step one, tidy the raw download of 925,000 rows "
  + "into about 14,000 clean ones. Step two, build 18 simple signals, including my "
  + "company-type mix. Step three, train four models: a conventional tree model, two neural "
  + "networks, and an anomaly detector that learns what normal looks like. Step four, produce "
  + "a risk score from zero to one and an explanation of why each case was flagged. The whole "
  + "thing is reproducible and runs end to end on a laptop with a single command.");

// ===================== 6. RESULTS (light) ===================== //
s = content(6, "Results", "It ranks the right places at the top");
s.addImage({ path: path.join(FIG, "roc_curves.png"), x: 6.95, y: 1.9, w: 5.65, h: 4.71 });
const stat = (y, big, lab) => {
  s.addText(big, { x: 0.7, y, w: 2.45, h: 0.85, fontFace: "Georgia", fontSize: 40, color: NAVY, bold: true, margin: 0 });
  s.addText(lab, { x: 3.25, y: y + 0.05, w: 3.5, h: 0.95, fontFace: "Calibri", fontSize: 14, color: INK, margin: 0, valign: "middle" });
};
stat(2.2, "0.94", "ranking quality on data the model never saw (1.0 is perfect)");
stat(3.5, "92%", "of the top 10% it flags really are known havens");
stat(4.8, "100%", "of its 50 highest-risk cases are known havens");
note(s, "Does it work? Yes. On data from a year the model never saw in training, it ranks "
  + "real havens above ordinary places very well, an AUC of 0.94 where 1.0 is perfect. More "
  + "usefully for a real audit team: if they only checked the top 10% it flags, 92% of those "
  + "would be genuine havens, and every one of its 50 highest-risk cases is a known haven. "
  + "The chart shows all four models, and they all do well.");

// ===================== 7. WHAT THE MODEL LEARNED (light) ===================== //
s = content(7, "Why it works", "The new signal does the heavy lifting");
s.addImage({ path: path.join(FIG, "shap_xgboost.png"), x: 6.75, y: 2.0, w: 5.85, h: 3.76 });
s.addText([
  { text: "This chart ranks which signals the model leans on most.", options: { bullet: true, breakLine: true } },
  { text: "The company-type signals (holding shells, real activity) come out on top.", options: { bullet: true, breakLine: true } },
  { text: "That is the original part of the project, and it pays off.", options: { bullet: true, breakLine: true } },
  { text: "Reassuring sign: the top-flagged places are Cayman, Luxembourg, Bermuda.", options: { bullet: true } },
], { x: 0.7, y: 2.2, w: 5.9, h: 4, fontFace: "Calibri", fontSize: 16, color: INK, paraSpaceAfter: 14 });
note(s, "Why does it work? This chart ranks which signals the model relies on most. The "
  + "company-type signals I added, like the share of empty holding shells versus real "
  + "activity, come out near the top. That is the original contribution of the project, and "
  + "it clearly pays off. As a sanity check, the places it scores highest are exactly the "
  + "ones you would expect: the Cayman Islands, Luxembourg, Bermuda.");

// ===================== 8. WHAT'S NEXT + CLOSE (dark) ===================== //
s = dark();
s.addShape(p.shapes.RECTANGLE, { x: 0.9, y: 0.7, w: 0.16, h: 0.16, fill: { color: GOLD } });
s.addText("HALFWAY POINT", { x: 1.18, y: 0.63, w: 8, h: 0.32, fontFace: "Calibri",
  fontSize: 12.5, color: GOLD, bold: true, charSpacing: 2, margin: 0 });
s.addText("An honest result, and where it goes next", { x: 0.88, y: 1.0, w: 11.5, h: 0.85,
  fontFace: "Georgia", fontSize: 28, color: WHITE, bold: true, margin: 0 });
s.addText([
  { text: "The simple tree model slightly beat my neural network, and a stats test says that gap is real, not luck. I report it straight.", options: { bullet: { indent: 16 }, breakLine: true, color: ICE } },
  { text: "Still to do for the final report: test how much the result leans on my list of known havens, and combine the models.", options: { bullet: { indent: 16 }, breakLine: true, color: ICE } },
  { text: "Tighten the evaluation and write it all up properly.", options: { bullet: { indent: 16 }, color: ICE } },
], { x: 1.1, y: 2.25, w: 11.2, h: 3.0, fontFace: "Calibri", fontSize: 18, paraSpaceAfter: 16 });
s.addShape(p.shapes.RECTANGLE, { x: 0.9, y: 5.95, w: 0.55, h: 0.14, fill: { color: GOLD } });
s.addText("Profit-shifting risk is learnable from public data. Thank you.", {
  x: 0.9, y: 6.2, w: 11.5, h: 0.6, fontFace: "Georgia", fontSize: 19, italic: true, color: GOLD, margin: 0 });
note(s, "To finish, an honest result and where this goes. The simple tree model slightly "
  + "beat my fancier neural network, and a statistical test says that gap is real rather than "
  + "luck, so I report it straight rather than overselling the neural side. For the final "
  + "report I will test how much the answer depends on my list of known havens, combine the "
  + "models, tighten the evaluation, and write it up in full. The headline holds: "
  + "profit-shifting risk really is learnable from public data. Thank you.");

p.writeFile({ fileName: path.join(__dirname, "..", "CM3070_Prototype_Demo.pptx") })
  .then(f => console.log("WROTE", f));
