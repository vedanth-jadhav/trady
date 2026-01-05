# Trady: The Complete Strategy Explained

## For Anyone Who Wants to Understand How This Works

---

## Table of Contents

1. [What is Polymarket?](#what-is-polymarket)
2. [The Big Idea: Following Smart Money](#the-big-idea)
3. [Who Are "Insiders"?](#who-are-insiders)
4. [How Do We Spot Them?](#how-do-we-spot-them)
5. [The Complete Detection System](#the-complete-detection-system)
6. [When Do We Trade?](#when-do-we-trade)
7. [How Much Do We Bet?](#how-much-do-we-bet)
8. [When Do We Exit?](#when-do-we-exit)
9. [Real Examples](#real-examples)
10. [Why This Works (Theory)](#why-this-works)
11. [Why This Might Not Work (Risks)](#risks)
12. [The Technical Flow](#technical-flow)

---

<a name="what-is-polymarket"></a>
## 1. What is Polymarket?

Polymarket is a **prediction market** — basically a betting platform for real-world events.

### How It Works

Imagine a market asking: **"Will Bitcoin hit $100,000 by December 2025?"**

- You can buy **"Yes"** shares if you think it will happen
- You can buy **"No"** shares if you think it won't
- Shares are priced between **$0.00 and $1.00**
- If the event happens, **"Yes" shares pay $1.00** each
- If it doesn't happen, **"No" shares pay $1.00** each

### Example

```
Market: "Will Bitcoin hit $100k by Dec 2025?"

Current prices:
  Yes: $0.35 (market thinks 35% chance)
  No:  $0.65 (market thinks 65% chance)

If you buy 100 "Yes" shares at $0.35 = You pay $35

Scenario A: Bitcoin DOES hit $100k
  → Your shares pay out $1.00 each
  → You receive $100
  → Profit: $100 - $35 = $65 (+186% return!)

Scenario B: Bitcoin does NOT hit $100k
  → Your shares are worth $0
  → You lose your $35 (-100% loss)
```

The key insight: **If you KNOW something will happen before others do, you can buy cheap and profit big.**

---

<a name="the-big-idea"></a>
## 2. The Big Idea: Following Smart Money

### The Core Concept

Some people have **information advantages**. They might know:
- A company is about to announce something
- A political decision has already been made behind closed doors
- An event outcome before it's public

These people trade on Polymarket. When they do, they leave **footprints**.

### Our Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   1. WATCH    →   2. DETECT    →   3. FOLLOW    →  4. WIN  │
│                                                             │
│   Monitor       Identify           Copy their      Profit   │
│   all trades    suspicious         trades          when     │
│   on Polymarket activity                           market   │
│                                                    resolves │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**We don't try to predict events ourselves. We find people who already know and follow them.**

### Why This Works (The Logic)

Think about it:

1. **If you're an insider**, you have valuable information
2. **You want to profit** from this information
3. **You need to trade** on Polymarket to make money
4. **When you trade**, you leave digital footprints
5. **We can see these footprints** and follow you

It's like watching where the smart money goes — and going there too.

---

<a name="who-are-insiders"></a>
## 3. Who Are "Insiders"?

### Types of Information Advantages

| Type | Example | How They Know |
|------|---------|---------------|
| **Corporate Insider** | Company employee | Knows earnings before announcement |
| **Political Insider** | Staffer, lobbyist | Knows policy decisions early |
| **Connected Individual** | Journalist, analyst | Has sources others don't |
| **Technical Insider** | Crypto developer | Knows about upcoming code changes |
| **Event Participant** | Sports referee, judge | Knows outcome before public |

### What Insider Trading Looks Like

Normal Trader:
```
- Has trading history
- Bets on many different markets
- Sometimes wins, sometimes loses
- Uses their main wallet
- Normal bet sizes
```

Suspicious "Insider":
```
- Brand new wallet (never seen before)
- Funded right before trading
- Makes ONE big bet
- Bets at very low odds (high risk, high reward)
- Only trades on specific obscure markets
- Exits only when market resolves (holds to the end)
```

### The Key Insight About Insider Behavior

> **"If someone TRULY knows an outcome, they don't exit early."**

Think about it:
- If you KNOW Bitcoin will hit $100k
- And you buy "Yes" at $0.20 (20% odds)
- Why would you sell at $0.90 and make 350% profit?
- You'd wait for $1.00 and make 400% profit!

**Real insiders hold until resolution. Retail traders get scared and exit early.**

This is a key signal we use: **Does this wallet hold to resolution, or panic-sell?**

---

<a name="how-do-we-spot-them"></a>
## 4. How Do We Spot Them?

We look for **patterns that don't make sense** for normal traders but DO make sense for insiders.

### The Five Detection Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │  WALLET   │  │   TRADE   │  │   MONEY   │  │   TIMING  │   │
│  │ FRESHNESS │  │   SIZE    │  │  SOURCE   │  │  PATTERN  │   │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘   │
│                                                                 │
│                      ┌───────────┐                              │
│                      │  WALLET   │                              │
│                      │ CLUSTERS  │                              │
│                      └───────────┘                              │
│                                                                 │
│         All signals combine into INSIDER LIKELIHOOD SCORE      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Let me explain each one:

---

### Signal 1: Wallet Freshness

**The Question:** Is this wallet suspiciously new?

**Why It Matters:**
Insiders don't want to be traced. They create NEW wallets specifically for insider trades. If a brand-new wallet suddenly appears and makes a big bet — that's suspicious.

**What We Check:**

| Check | Suspicious If... | Why |
|-------|------------------|-----|
| First trade ever? | Yes, this is their first | Insiders create fresh wallets |
| New to Polymarket? | Never traded here before | Even if wallet is old, new to betting = suspicious |
| Recently funded? | Got money in last 24 hours | Funded right before trading = planned |
| Low trade count? | Less than 5 trades total | Not enough history to establish pattern |

**Example:**

```
🚨 SUSPICIOUS:
Wallet 0x7a3f...
  - Created: 2 hours ago
  - First-ever Polymarket trade
  - Funded 30 minutes before trade
  - Placed $50,000 bet on obscure political market

vs.

✅ NORMAL:
Wallet 0x9b2c...
  - Trading on Polymarket for 14 months
  - 347 previous trades
  - Regular funding pattern
  - Placed $50,000 bet (consistent with history)
```

---

### Signal 2: Trade Size

**The Question:** Is this trade unusually large?

**Why It Matters:**
If you KNOW something will happen, you bet big. Retail traders make small, diversified bets. Insiders concentrate their bets.

**How We Measure "Large":**

We use **RELATIVE sizing**, not absolute. $10,000 might be huge in a small market but tiny in a large one.

```
Trade Size Score = Your Trade Size / Typical Trade Size in This Market

Example:
  Market "Will X happen?" has average trade of $500
  Someone trades $15,000
  Score = $15,000 / $500 = 30x normal = VERY SUSPICIOUS
```

**What We Check:**

| Metric | Calculation | Suspicious If... |
|--------|-------------|------------------|
| % of market volume | trade / total market volume | > 5% of entire market |
| vs. market median | trade / median trade size | > 10x median |
| vs. own history | trade / wallet's average | > 3x their normal |
| position concentration | total in this market / portfolio | All eggs in one basket |

**Example:**

```
Market: "Will Company X announce merger by Friday?"
  - Total market volume: $200,000
  - Median trade size: $150
  - Average trade size: $400

🚨 SUSPICIOUS TRADE:
  - Wallet 0x3f2a...
  - Trade: $25,000 (12.5% of entire market!)
  - This is 166x the median
  - Wallet has no other positions

✅ NORMAL TRADE:
  - Wallet 0x8c1b...
  - Trade: $500
  - 0.25% of market, 3x median
  - Has 12 other positions
```

---

### Signal 3: Money Source (Funding)

**The Question:** Where did the money come from?

**Why It Matters:**
Insiders don't want to be traced. They use privacy tools, bridge funds from other chains, or use fresh exchange withdrawals.

**Funding Sources Ranked by Suspicion:**

| Source | Suspicion Level | Why |
|--------|-----------------|-----|
| **Privacy tools (Tornado Cash)** | 🔴 CRITICAL | Deliberately hiding money trail |
| **From another wallet we're tracking** | 🟠 HIGH | Network of connected wallets |
| **Cross-chain bridge** | 🟡 MEDIUM-HIGH | Obfuscating source chain |
| **Exchange withdrawal** | 🟡 MEDIUM | Could be anyone, hard to trace |
| **Known wallet (long history)** | 🟢 LOW | Normal funding pattern |

**Example:**

```
🚨 CRITICAL SUSPICION:
Wallet 0x7f9a... received $30,000
  Source: Tornado Cash (privacy mixer)
  Then immediately bet $29,500 on "CEO resignation" market
  → This person is HIDING their identity deliberately

🟠 HIGH SUSPICION:
Wallet 0x4b2c... received $15,000
  Source: Wallet 0x8a1f... (which we flagged yesterday)
  → These wallets are probably the same person

✅ LOW SUSPICION:
Wallet 0x2d5e... received $10,000
  Source: Same Coinbase account they've used for 2 years
  → Normal funding behavior
```

---

### Signal 4: Timing Patterns

**The Question:** WHEN did they trade?

**Why It Matters:**
Insiders often trade at specific times — right before news, during off-hours when fewer eyes are watching, or in rapid bursts.

**Timing Signals:**

| Pattern | What It Means | Suspicion Level |
|---------|---------------|-----------------|
| **Pre-news** | Trade happens 1-24 hours before major announcement | 🔴 HIGH |
| **Off-hours** | Trade at 3am when retail is sleeping | 🟠 MEDIUM |
| **Before price spike** | Trade right before the market moves big | 🔴 HIGH |
| **Rapid succession** | Multiple trades in 10 minutes | 🟡 MEDIUM |

**Example:**

```
Timeline of suspicious trading:

Monday 11:00pm: CEO resigns (becomes public)
                ↑
Monday 9:30pm:  Wallet 0x3a1f... buys $40k "Yes"
                ↑
Monday 9:00pm:  Wallet receives $42k from Tornado Cash
                ↑
Monday 8:30pm:  Wallet created

🚨 PATTERN: Fresh wallet → Private funding → Large trade → 90 minutes later news breaks

This is textbook insider trading behavior.
```

---

### Signal 5: Wallet Clusters

**The Question:** Is this wallet connected to other suspicious wallets?

**Why It Matters:**
Smart insiders spread their bets across multiple wallets to avoid detection. But they leave patterns that connect these wallets.

**How Wallets Get Clustered:**

```
Method 1: FUNDING SOURCE
  Wallet A ←── $10,000 ──┐
  Wallet B ←── $10,000 ──┼── Same source wallet
  Wallet C ←── $10,000 ──┘

  → These are probably the same person

Method 2: BEHAVIORAL
  Wallet A trades at 2:14am, 2:17am, 2:21am
  Wallet B trades at 2:15am, 2:18am, 2:22am

  → Trading at same times = probably coordinated

Method 3: POSITION CORRELATION
  Wallet A buys "Yes" on Market X
  Wallet B buys "Yes" on Market X (2 min later)
  Wallet A buys "No" on Market Y
  Wallet B buys "No" on Market Y (1 min later)

  → Positions are correlated = probably same person
```

**Why Clusters Matter:**

If we identify a CLUSTER of wallets:
- We count their COMBINED activity
- $10k × 5 wallets = $50k insider position
- This is MORE suspicious than one $50k trade (they're hiding!)

---

<a name="the-complete-detection-system"></a>
## 5. The Complete Detection System

Now let's see how ALL signals combine:

### Step 1: Calculate Individual Scores

For each trade, we calculate:

```python
Freshness Score:    0.0 to 1.0  (how new is wallet?)
Timing Score:       0.0 to 1.0  (suspicious timing?)
Sizing Score:       0.0 to 1.0  (unusual size?)
Funding Score:      0.0 to 1.0  (suspicious source?)
Cluster Score:      0.0 to 1.0  (part of suspicious group?)
```

### Step 2: Weight and Combine

Each category has a weight based on importance:

```
                 SIGNAL WEIGHTS
┌────────────────────────────────────────┐
│  Freshness:     25%  ████████████░░░░  │
│  Funding:       25%  ████████████░░░░  │
│  Timing:        20%  ██████████░░░░░░  │
│  Sizing:        20%  ██████████░░░░░░  │
│  Clustering:    10%  █████░░░░░░░░░░░  │
└────────────────────────────────────────┘
```

### Step 3: Apply Negative Signals

Some things DECREASE insider likelihood:

```
Negative Signals (reduce score):
  - Long trading history (6+ months)     → -30%
  - Known public entity (fund, MM)       → -20%
  - Retail behavior (many small bets)    → -30%
  - Frequently exits early               → -20%
```

### Step 4: Apply Market Context

Some markets have MORE insider potential:

```
High-Insider Markets (boost score):
  - Political events (+20%)
  - Corporate announcements (+20%)
  - Crypto governance (+15%)

Low-Insider Markets (reduce score):
  - Sports (-20%)
  - Weather (-30%)
  - Random events (-20%)
```

### Step 5: Final Score Calculation

```
Raw Score = (Freshness × 0.25) + (Funding × 0.25) +
            (Timing × 0.20) + (Sizing × 0.20) + (Cluster × 0.10)

Adjusted Score = Raw Score × (1 - Negative Signals × 0.5)

Final Score = Adjusted Score × Market Boost

Example:
  Freshness: 0.9, Funding: 0.8, Timing: 0.7, Sizing: 0.6, Cluster: 0.5
  Raw = (0.9×0.25) + (0.8×0.25) + (0.7×0.20) + (0.6×0.20) + (0.5×0.10)
      = 0.225 + 0.200 + 0.140 + 0.120 + 0.050
      = 0.735

  Negative signals: 0.1 (some history)
  Adjusted = 0.735 × (1 - 0.1×0.5) = 0.735 × 0.95 = 0.698

  Market boost: Political market = 1.2
  Final = 0.698 × 1.2 = 0.838

  → 83.8% INSIDER LIKELIHOOD (HIGH CONFIDENCE)
```

### The Machine Learning Layer

On top of this, we train an **XGBoost model** that:

1. Learns from historical data (which signals led to wins?)
2. Finds patterns humans might miss
3. Calibrates probabilities more accurately
4. Adjusts weights based on what actually works

Think of it as:
- **Rule-based system** = Human expert knowledge
- **ML model** = Learning from thousands of historical examples
- **Combined** = Best of both worlds

---

<a name="when-do-we-trade"></a>
## 6. When Do We Trade?

### Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRADE DECISION TREE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Suspicious trade detected                                     │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────────┐                                     │
│   │ Final Score >= 0.6?  │                                     │
│   └──────────┬───────────┘                                     │
│              │                                                  │
│      NO ─────┼───── YES                                        │
│      │              │                                           │
│      ▼              ▼                                           │
│   IGNORE     ┌──────────────────────┐                          │
│              │ Risk limits OK?       │                          │
│              │ - Portfolio exposure  │                          │
│              │ - Position limits     │                          │
│              │ - Wallet limits       │                          │
│              └──────────┬───────────┘                          │
│                         │                                       │
│                 NO ─────┼───── YES                             │
│                 │              │                                │
│                 ▼              ▼                                │
│              SKIP       ┌──────────────────────┐               │
│                         │ Already have position?│               │
│                         └──────────┬───────────┘               │
│                                    │                            │
│                            YES ────┼──── NO                    │
│                            │              │                     │
│                            ▼              ▼                     │
│                         SKIP        EXECUTE TRADE               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Confidence Tiers

| Tier | Score Range | Action |
|------|-------------|--------|
| **Very High** | 0.80 - 1.00 | Trade with maximum size |
| **High** | 0.60 - 0.79 | Trade with normal size |
| **Medium** | 0.40 - 0.59 | Skip (too risky) |
| **Low** | 0.20 - 0.39 | Skip |
| **Very Low** | 0.00 - 0.19 | Definitely skip |

**We only trade on HIGH and VERY HIGH confidence signals.**

---

<a name="how-much-do-we-bet"></a>
## 7. How Much Do We Bet?

### Position Sizing Philosophy

> "Size your bets based on confidence, not greed."

### The Formula

```
Position Size = Base Size × Confidence Multiplier

Where:
  Base Size = $100 (configurable)
  Confidence Multiplier = 1 + (score - 0.5) × 2 × (max_mult - 1)
  max_mult = 5 (maximum 5x base for very high confidence)

Example calculations:

Score 0.60 (barely high):
  Multiplier = 1 + (0.60 - 0.5) × 2 × 4 = 1 + 0.8 = 1.8
  Size = $100 × 1.8 = $180

Score 0.80 (very high):
  Multiplier = 1 + (0.80 - 0.5) × 2 × 4 = 1 + 2.4 = 3.4
  Size = $100 × 3.4 = $340

Score 0.95 (extremely high):
  Multiplier = 1 + (0.95 - 0.5) × 2 × 4 = 1 + 3.6 = 4.6
  Size = $100 × 4.6 = $460
```

### Risk Limits (Hard Caps)

```
MULTI-LAYER PROTECTION
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Layer 1: PORTFOLIO                                 │
│  └── Max 80% of capital in positions at once       │
│      ($10k capital → max $8k exposed)              │
│                                                     │
│  Layer 2: PER MARKET                               │
│  └── Max $1,000 in any single market               │
│      (prevents concentration risk)                  │
│                                                     │
│  Layer 3: PER WALLET                               │
│  └── Max $2,000 following any single wallet        │
│      (don't over-trust one signal source)          │
│                                                     │
│  Layer 4: CONCURRENT POSITIONS                      │
│  └── Max 50 open positions at once                 │
│      (forces diversification)                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

<a name="when-do-we-exit"></a>
## 8. When Do We Exit?

### The Core Strategy: HOLD TO RESOLUTION

This is the key insight:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "If an insider KNOWS the outcome, they don't exit early."    │
│                                                                 │
│   Example:                                                      │
│   - Insider buys "Yes" at $0.20                                 │
│   - They KNOW the event will happen                             │
│   - Price rises to $0.90                                        │
│   - Normal person might sell (+350% profit)                     │
│   - But insider KNOWS it goes to $1.00                          │
│   - So insider holds (+400% profit)                             │
│                                                                 │
│   WE COPY THIS BEHAVIOR: HOLD UNTIL THE MARKET RESOLVES        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Exit Scenarios

| Scenario | What Happens | Our Action |
|----------|--------------|------------|
| Market resolves in our favor | "Yes" becomes $1.00 | Collect full payout |
| Market resolves against us | "Yes" becomes $0.00 | Lose investment |
| Market still open | Waiting | Continue holding |

**We do NOT:**
- Panic sell when price drops
- Take profit at 90%
- Exit based on time

**We DO:**
- Wait for official resolution
- Accept the binary outcome
- Trust our signal detection

---

<a name="real-examples"></a>
## 9. Real Examples

### Example 1: The CEO Resignation

```
MARKET: "Will TechCorp CEO resign by end of month?"
Current Price: Yes = $0.15 (market thinks unlikely)

WHAT WE DETECT:

Tuesday 2:00pm:
  New wallet 0x7a3f... created

Tuesday 2:15pm:
  Wallet receives $100,000 from Tornado Cash (privacy!)

Tuesday 2:30pm:
  Wallet buys $95,000 of "Yes" shares at $0.15

OUR ANALYSIS:
  Freshness:  0.95 (brand new wallet)
  Funding:    1.00 (privacy tool = CRITICAL)
  Sizing:     0.90 (huge relative to market)
  Timing:     0.70 (middle of day, but rapid setup)

  Raw Score: 0.91
  Final Score: 0.89 (VERY HIGH)

OUR ACTION:
  Buy "Yes" shares at $0.16 (slightly worse due to impact)
  Size: $400 (max for this confidence)

OUTCOME:
  Wednesday 9:00am: CEO resignation announced
  "Yes" resolves to $1.00

  Our P&L:
    Bought: $400 at $0.16 = 2,500 shares
    Payout: 2,500 × $1.00 = $2,500
    Profit: $2,500 - $400 = $2,100 (+525%)
```

### Example 2: The False Positive

```
MARKET: "Will Bitcoin ETF be approved this week?"
Current Price: Yes = $0.40

WHAT WE DETECT:

Thursday 11:00pm:
  Wallet 0x9c2b... (existing, but quiet for 2 months)
  Buys $30,000 of "Yes" at $0.40

OUR ANALYSIS:
  Freshness:  0.40 (not new, but inactive period)
  Funding:    0.30 (normal exchange withdrawal)
  Sizing:     0.60 (large but not extreme)
  Timing:     0.50 (off-hours, but not unusual for crypto)

  But also:
  Negative: Wallet has 8-month history = -20%
  Negative: Has traded this market before = -10%

  Final Score: 0.42 (MEDIUM - below threshold)

OUR ACTION:
  DO NOT TRADE (score below 0.60 threshold)

OUTCOME:
  ETF not approved this week
  "Yes" resolves to $0.00

  Our P&L: $0 (we didn't trade!)

  The signal was a wealthy retail trader making a hopeful bet,
  not an insider. Our filters correctly avoided it.
```

### Example 3: The Cluster Detection

```
MARKET: "Will Merger X be announced?"
Current Price: Yes = $0.25

WHAT WE DETECT:

Monday:
  Wallet A: Buys $5,000 "Yes"
  Wallet B: Buys $5,000 "Yes" (2 min later)
  Wallet C: Buys $5,000 "Yes" (3 min later)

OUR CLUSTER ANALYSIS:
  - All 3 wallets funded from same source 1 hour ago
  - All 3 are new wallets (created today)
  - All 3 trade within 5 minutes
  - All 3 same position size

  CLUSTER DETECTED: Wallets A, B, C
  Combined position: $15,000

INDIVIDUAL ANALYSIS (for each):
  Freshness:  0.85
  Funding:    0.70 (traceable but suspicious pattern)
  Sizing:     0.50 (each trade is moderate)
  Timing:     0.80 (coordinated timing)
  Cluster:    0.90 (high coordination)

  Final Score: 0.78 (HIGH)

OUR ACTION:
  Buy "Yes" at $0.26
  Size: $300

OUTCOME:
  Merger announced Thursday
  "Yes" resolves to $1.00
  Profit: $1,150 - $300 = $850 (+283%)
```

---

<a name="why-this-works"></a>
## 10. Why This Works (Theory)

### The Information Asymmetry Edge

```
                    INFORMATION FLOW

Corporate Insider ──────► Polymarket Trade ──────► Our Detection
      │                         │                        │
      │                         │                        │
   KNOWS                    LEAVES                    WE SEE
   outcome                  footprint                 footprint
                                                          │
                                                          ▼
                                                    WE FOLLOW
```

### Why Insiders Can't Hide

| They Try To... | But They Can't Because... |
|----------------|---------------------------|
| Use new wallets | We detect freshness |
| Trade small amounts | They need profit, so they trade big |
| Mix with other trades | Pattern still visible in aggregate |
| Use privacy tools | We specifically look for this! |
| Spread across wallets | We cluster related wallets |
| Trade at random times | Timing relative to news still shows |

### The Mathematical Edge

```
Expected Value = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

Normal trading (no edge):
  EV = (50% × $100) - (50% × $100) = $0

Our strategy (with edge):
  If we achieve 65% win rate on filtered signals:
  EV = (65% × $200) - (35% × $100) = $130 - $35 = $95 per trade

  With position sizing:
  - 100 trades per month
  - $95 expected per trade
  - Expected monthly: $9,500
```

---

<a name="risks"></a>
## 11. Why This Might Not Work (Risks)

### Risk 1: False Positives

Not every suspicious pattern is an insider. Some are:
- Wealthy retail traders
- Algorithmic traders
- Market makers

**Mitigation:** High confidence threshold (0.60+), negative signal filters

### Risk 2: Insiders Are Wrong

Sometimes insiders have bad information:
- Their source was wrong
- Plans changed after they traded
- They misunderstood what they heard

**Mitigation:** Diversification across many signals

### Risk 3: Detection Becomes Known

If insiders learn we're tracking them:
- They might change patterns
- They might create honeypot traps
- The edge could disappear

**Mitigation:** Continuous model retraining, pattern evolution

### Risk 4: Markets Don't Resolve

Some markets:
- Get cancelled
- Have disputed resolutions
- Take very long to resolve

**Mitigation:** Only trade active markets with clear resolution criteria

### Risk 5: Model Overfitting

Our ML model might:
- Learn patterns that don't generalize
- Perform well on backtest, poorly live

**Mitigation:** Proper train/test splits, out-of-sample validation

---

<a name="technical-flow"></a>
## 12. The Technical Flow

### End-to-End Process

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRADY DATA FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: DATA COLLECTION                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Polymarket API                                           │  │
│  │       │                                                   │  │
│  │       ▼                                                   │  │
│  │  Fetch all trades (90 days, 100 markets)                 │  │
│  │       │                                                   │  │
│  │       ▼                                                   │  │
│  │  Store in Parquet files                                  │  │
│  │  (trades.parquet, markets.parquet, wallets.parquet)      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  PHASE 2: WALLET PROFILING                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  For each wallet:                                        │  │
│  │    - Calculate freshness (how new?)                      │  │
│  │    - Analyze funding (where from?)                       │  │
│  │    - Profile behavior (trading patterns)                 │  │
│  │    - Find clusters (related wallets)                     │  │
│  │                                                          │  │
│  │  Output: wallet_profiles.parquet                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  PHASE 3: SIGNAL DETECTION                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  For each trade:                                         │  │
│  │    - Run all 5 detectors                                 │  │
│  │    - Apply negative filters                              │  │
│  │    - Consider market context                             │  │
│  │    - Calculate raw score                                 │  │
│  │                                                          │  │
│  │  Output: signals.parquet                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  PHASE 4: ML SCORING                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  - Extract 50+ features per trade                        │  │
│  │  - Run through trained XGBoost ensemble                  │  │
│  │  - Calibrate probabilities                               │  │
│  │  - Apply rule-based overrides                            │  │
│  │                                                          │  │
│  │  Output: scored_trades.parquet (with final_score)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  PHASE 5: BACKTEST                                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  - Filter to high-confidence signals (≥0.60)            │  │
│  │  - Simulate trades with position sizing                  │  │
│  │  - Track portfolio through time                          │  │
│  │  - Wait for market resolutions                           │  │
│  │  - Calculate P&L, metrics                                │  │
│  │                                                          │  │
│  │  Output: backtest_results/                               │  │
│  │    - trade_log.parquet                                   │  │
│  │    - equity_curve.parquet                                │  │
│  │    - metrics.json                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  PHASE 6: VISUALIZATION                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Terminal Dashboard showing:                             │  │
│  │    - Performance metrics                                 │  │
│  │    - Signal feed                                         │  │
│  │    - Open positions                                      │  │
│  │    - Equity curve                                        │  │
│  │    - Trade history                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary: The One-Page Version

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    TRADY IN 30 SECONDS                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WHAT:    A bot that detects and follows "insider" trading     │
│           on Polymarket prediction markets                      │
│                                                                 │
│  HOW:     1. Watch all trades on Polymarket                    │
│           2. Score each trade for "insider likelihood"          │
│           3. Follow high-confidence signals (≥60%)             │
│           4. Size bets based on confidence                      │
│           5. Hold until market resolves                         │
│                                                                 │
│  SIGNALS: Fresh wallets + Privacy funding + Big trades +       │
│           Suspicious timing + Wallet clusters                   │
│           MINUS: Long history, retail behavior, early exits     │
│                                                                 │
│  WHY IT   Insiders MUST trade to profit. When they trade,      │
│  WORKS:   they leave footprints. We detect and follow.         │
│                                                                 │
│  EDGE:    If we achieve 65% win rate vs 50% random:            │
│           Expected value = +$95 per $100 risked                 │
│                                                                 │
│  RISK:    False positives, wrong insiders, pattern changes,    │
│           model overfitting, market not resolving               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0*
*Created: January 2025*
*Project: Trady - Polymarket Insider Detection Bot*
