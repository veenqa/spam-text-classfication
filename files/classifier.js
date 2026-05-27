/**
 * classifier.js
 * Heuristic spam scorer using weighted token matching.
 * In production, replace scoreMessage() with a backend API
 * call to the Python scikit-learn MLP pipeline (spam_classifier.py).
 */

const SPAM_TOKENS = [
  { t: 'won',              w: 2.0 },
  { t: 'winner',           w: 2.0 },
  { t: 'free',             w: 1.5 },
  { t: 'prize',            w: 2.0 },
  { t: 'claim',            w: 2.0 },
  { t: 'congratulations',  w: 2.0 },
  { t: 'urgent',           w: 2.0 },
  { t: 'click here',       w: 2.0 },
  { t: 'guaranteed',       w: 2.0 },
  { t: 'limited time',     w: 1.5 },
  { t: 'earn',             w: 1.5 },
  { t: 'lottery',          w: 2.5 },
  { t: 'selected',         w: 1.5 },
  { t: 'verify',           w: 1.5 },
  { t: 'suspended',        w: 2.0 },
  { t: 'account suspended',w: 2.5 },
  { t: 'charged',          w: 1.5 },
  { t: 'buy now',          w: 2.0 },
  { t: 'cheap',            w: 1.0 },
  { t: 'replica',          w: 2.0 },
  { t: 'virus',            w: 2.0 },
  { t: 'weight loss',      w: 1.5 },
  { t: 'pill',             w: 1.5 },
  { t: 'bitcoin',          w: 1.5 },
  { t: 'crypto',           w: 1.5 },
  { t: 'investment',       w: 1.0 },
  { t: 'returns',          w: 1.0 },
  { t: 'bank details',     w: 2.5 },
  { t: 'insurance',        w: 1.0 },
  { t: 'refinance',        w: 1.5 },
  { t: 'survey',           w: 1.0 },
  { t: 'gift card',        w: 2.0 },
  { t: 'paytm',            w: 1.0 },
  { t: 'phonepay',         w: 1.0 },
  { t: 'upi',              w: 1.0 },
  { t: 'kyc',              w: 1.5 },
  { t: 'aadhaar',          w: 1.0 },
  { t: 'pan card',         w: 1.0 },
  { t: 'lakh',             w: 1.5 },
  { t: 'crore',            w: 1.5 },
  { t: 'work from home',   w: 2.0 },
  { t: 'no experience',    w: 2.0 },
  { t: 'get rich',         w: 2.0 },
  { t: 'make money',       w: 2.0 },
  { t: 'fast cash',        w: 2.0 },
  { t: 'kbc',              w: 2.0 },
  { t: 'lic',              w: 1.0 },
  { t: 'sbi alert',        w: 2.0 },
  { t: 'otp',              w: 1.5 },
  { t: 'exclusive',        w: 1.0 },
  { t: 'offer',            w: 1.0 },
];

const HAM_TOKENS = [
  { t: 'meeting',        w: 2.0 },
  { t: 'report',         w: 2.0 },
  { t: 'attached',       w: 1.5 },
  { t: 'schedule',       w: 1.5 },
  { t: 'review',         w: 1.5 },
  { t: 'project',        w: 1.5 },
  { t: 'team',           w: 1.5 },
  { t: 'please',         w: 1.0 },
  { t: 'thanks',         w: 1.5 },
  { t: 'follow up',      w: 2.0 },
  { t: 'deadline',       w: 2.0 },
  { t: 'budget',         w: 1.5 },
  { t: 'confirmed',      w: 1.5 },
  { t: 'appointment',    w: 2.0 },
  { t: 'delivery',       w: 1.5 },
  { t: 'shipped',        w: 1.5 },
  { t: 'invoice',        w: 2.0 },
  { t: 'compliance',     w: 2.0 },
  { t: 'training',       w: 1.0 },
  { t: 'rescheduled',    w: 2.0 },
  { t: 'standup',        w: 2.0 },
  { t: 'quarterly',      w: 2.0 },
  { t: 'deployed',       w: 2.0 },
  { t: 'production',     w: 1.5 },
  { t: 'application',    w: 1.0 },
  { t: 'conference',     w: 2.0 },
  { t: 'namaste',        w: 1.5 },
  { t: 'regards',        w: 1.5 },
  { t: 'hi',             w: 0.5 },
  { t: 'hello',          w: 0.5 },
];

/**
 * Scores a raw message and returns spam/ham probabilities
 * plus the matched signal tokens for display.
 *
 * @param {string} text  Raw message text
 * @returns {{ spamProb: number, hamProb: number, foundSpam: string[], foundHam: string[] }}
 */
function scoreMessage(text) {
  const lower = text.toLowerCase();
  let spamScore = 0;
  let hamScore  = 0;
  const foundSpam = [];
  const foundHam  = [];

  SPAM_TOKENS.forEach(({ t, w }) => {
    if (lower.includes(t)) {
      spamScore += w;
      foundSpam.push(t);
    }
  });

  HAM_TOKENS.forEach(({ t, w }) => {
    if (lower.includes(t)) {
      hamScore += w;
      foundHam.push(t);
    }
  });

  // Extra heuristic signals
  if (/[A-Z]{3,}/.test(text))                     spamScore += 2.0;   // ALL CAPS words
  if (/[\u20b9\$£€]/.test(text))                   spamScore += 1.5;   // currency symbols
  if (/!{2,}/.test(text))                           spamScore += 1.0;   // multiple !!
  if (/\d{1,2},\d{2},\d{3}/.test(text))            spamScore += 1.5;   // Indian number format ₹X,XX,XXX
  if (/\d+%/.test(text))                            spamScore += 0.5;   // percentage claims

  const total = spamScore + hamScore + 0.5;
  let spamProb = Math.min(0.97, Math.max(0.03, spamScore / total));

  // Anchor extremes when one side is silent
  if (spamScore === 0 && hamScore > 0) spamProb = 0.05;
  if (hamScore  === 0 && spamScore > 0) spamProb = 0.93;

  return {
    spamProb,
    hamProb: 1 - spamProb,
    foundSpam,
    foundHam,
  };
}
