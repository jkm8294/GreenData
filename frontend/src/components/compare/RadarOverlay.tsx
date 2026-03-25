"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import { Score } from "@/lib/api";

const COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7"];

const DIMENSIONS = [
  { key: "power_score", label: "Power" },
  { key: "environmental_score", label: "Environmental" },
  { key: "social_score", label: "Social" },
  { key: "psi", label: "PSI" },
  { key: "confidence", label: "Confidence" },
];

export default function RadarOverlay({ locations }: { locations: Score[] }) {
  const data = DIMENSIONS.map((dim) => {
    const point: Record<string, string | number> = { dimension: dim.label };
    locations.forEach((loc) => {
      const val = (loc as unknown as Record<string, number>)[dim.key];
      point[loc.county_name || loc.county_fips] = val != null ? Math.round(val) : 0;
    });
    return point;
  });

  return (
    <ResponsiveContainer width="100%" height={400}>
      <RadarChart data={data}>
        <PolarGrid stroke="#2a2b35" />
        <PolarAngleAxis dataKey="dimension" tick={{ fill: "#a1a1aa", fontSize: 12 }} />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tick={{ fill: "#52525b", fontSize: 10 }}
        />
        {locations.map((loc, i) => (
          <Radar
            key={loc.county_fips}
            name={loc.county_name || loc.county_fips}
            dataKey={loc.county_name || loc.county_fips}
            stroke={COLORS[i % COLORS.length]}
            fill={COLORS[i % COLORS.length]}
            fillOpacity={0.15}
            strokeWidth={2}
          />
        ))}
        <Legend wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }} />
        <Tooltip
          contentStyle={{
            background: "#1a1b23",
            border: "1px solid #2a2b35",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
