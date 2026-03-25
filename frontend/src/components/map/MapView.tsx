"use client";

import { useEffect, useRef } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { Score } from "@/lib/api";
import { psiColor } from "@/lib/utils";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || "";

interface Props {
  scores: Score[];
  onSelect?: (fips: string) => void;
}

export default function MapView({ scores, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    if (!MAPBOX_TOKEN) return;

    mapboxgl.accessToken = MAPBOX_TOKEN;

    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: "mapbox://styles/mapbox/dark-v11",
      center: [-98.5, 39.8],
      zoom: 3.8,
    });

    map.addControl(new mapboxgl.NavigationControl(), "top-right");
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Add/update markers when scores change
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !scores.length) return;

    // Wait for style to load
    const addMarkers = () => {
      // Remove existing markers
      document.querySelectorAll(".gd-marker").forEach((el) => el.remove());

      scores.forEach((s) => {
        if (!s.lat || !s.lon) return;

        const color = psiColor(s.psi);
        const el = document.createElement("div");
        el.className = "gd-marker";
        el.style.cssText = `
          width: 16px; height: 16px; border-radius: 50%;
          background: ${color}; border: 2px solid ${color}40;
          cursor: pointer; box-shadow: 0 0 8px ${color}60;
          transition: transform 0.15s;
        `;
        el.addEventListener("mouseenter", () => {
          el.style.transform = "scale(1.4)";
        });
        el.addEventListener("mouseleave", () => {
          el.style.transform = "scale(1)";
        });

        const popup = new mapboxgl.Popup({ offset: 12, closeButton: false }).setHTML(`
          <div style="font-family: system-ui; font-size: 13px;">
            <div style="font-weight: 600;">${s.county_name || s.county_fips}</div>
            <div style="color: #a1a1aa; font-size: 11px;">${s.state || ""}</div>
            <div style="margin-top: 6px; font-size: 20px; font-weight: 700; color: ${color};">
              ${s.psi.toFixed(1)} <span style="font-size: 11px; color: #a1a1aa;">PSI</span>
            </div>
            <div style="margin-top: 4px; font-size: 11px; color: #a1a1aa;">
              PWR ${s.power_score?.toFixed(0) ?? "—"} &middot;
              ENV ${s.environmental_score?.toFixed(0) ?? "—"} &middot;
              SOC ${s.social_score?.toFixed(0) ?? "—"}
            </div>
          </div>
        `);

        new mapboxgl.Marker({ element: el })
          .setLngLat([s.lon, s.lat])
          .setPopup(popup)
          .addTo(map);

        el.addEventListener("click", () => {
          onSelect?.(s.county_fips);
        });
      });
    };

    if (map.isStyleLoaded()) {
      addMarkers();
    } else {
      map.on("load", addMarkers);
    }
  }, [scores, onSelect]);

  if (!MAPBOX_TOKEN) {
    return (
      <div className="flex h-full items-center justify-center rounded-lg border border-[var(--card-border)] bg-[var(--card)] text-[var(--muted)] text-sm">
        Set NEXT_PUBLIC_MAPBOX_TOKEN in .env.local to enable the map
      </div>
    );
  }

  return <div ref={containerRef} className="h-full w-full rounded-lg" />;
}
