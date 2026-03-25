/** PSI color: green (70+), yellow (50–69), red (<50) */
export function psiColor(psi: number): string {
  if (psi >= 70) return "#22c55e";
  if (psi >= 50) return "#eab308";
  return "#ef4444";
}

/** PSI label */
export function psiLabel(psi: number): string {
  if (psi >= 70) return "High Suitability";
  if (psi >= 50) return "Moderate";
  return "Low Suitability";
}

/** Format a number for display */
export function fmt(n: number | null | undefined, decimals = 1): string {
  if (n === null || n === undefined) return "N/A";
  return n.toFixed(decimals);
}

/** FIPS to state abbreviation (2-digit prefix) */
const FIPS_TO_STATE: Record<string, string> = {
  "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
  "08": "CO", "09": "CT", "10": "DE", "12": "FL", "13": "GA",
  "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
  "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
  "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO",
  "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ",
  "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
  "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC",
  "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT",
  "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
};

export function fipsToState(fips: string): string {
  return FIPS_TO_STATE[fips.slice(0, 2)] || fips.slice(0, 2);
}
