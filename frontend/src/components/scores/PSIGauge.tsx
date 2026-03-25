"use client";

import { psiColor, psiLabel } from "@/lib/utils";

export default function PSIGauge({ score }: { score: number }) {
  const color = psiColor(score);
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const filled = (score / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="140" height="140" viewBox="0 0 140 140">
        {/* Background circle */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke="var(--card-border)"
          strokeWidth="10"
        />
        {/* Filled arc */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeDasharray={`${filled} ${circumference}`}
          strokeDashoffset={circumference * 0.25}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
          className="transition-all duration-700"
        />
        {/* Center text */}
        <text x="70" y="66" textAnchor="middle" fill={color} fontSize="28" fontWeight="bold">
          {Math.round(score)}
        </text>
        <text x="70" y="86" textAnchor="middle" fill="var(--muted)" fontSize="11">
          PSI
        </text>
      </svg>
      <span className="text-xs font-medium" style={{ color }}>
        {psiLabel(score)}
      </span>
    </div>
  );
}
