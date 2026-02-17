"use client";

import { useEffect, useState } from "react";
import { IconMoon, IconSunHigh } from "./Icons";

export default function ThemeToggle() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const toggleTheme = () => {
    const html = document.documentElement;
    const current = html.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);

    // Update header/menu CSS variables
    html.style.setProperty(
      "--header-bg",
      next === "dark"
        ? "rgba(13,17,23,0.92)"
        : "rgba(250,251,253,0.92)"
    );
    html.style.setProperty(
      "--menu-bg",
      next === "dark"
        ? "rgba(13,17,23,0.95)"
        : "rgba(250,251,253,0.95)"
    );

    const body = document.body;
    if (body) {
      // Wait for theme to apply then update meta theme-color
      requestAnimationFrame(() => {
        const bgColor = window.getComputedStyle(body).backgroundColor;
        document
          .querySelector("meta[name='theme-color']")
          ?.setAttribute("content", bgColor);
      });
    }
  };

  if (!mounted) {
    return (
      <button
        className="focus-outline relative size-12 p-4 sm:size-8 hover:[&>svg]:stroke-accent"
        title="Toggles light & dark"
        aria-label="auto"
      >
        <IconMoon className="absolute top-[50%] left-[50%] -translate-[50%] scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
        <IconSunHigh className="absolute top-[50%] left-[50%] -translate-[50%] scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
      </button>
    );
  }

  return (
    <button
      onClick={toggleTheme}
      className="focus-outline relative size-12 p-4 sm:size-8 hover:[&>svg]:stroke-accent"
      title="Toggles light & dark"
      aria-label="auto"
      aria-live="polite"
    >
      <IconMoon className="absolute top-[50%] left-[50%] -translate-[50%] scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
      <IconSunHigh className="absolute top-[50%] left-[50%] -translate-[50%] scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
    </button>
  );
}
