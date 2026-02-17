"use client";

import { useEffect, useRef, useState } from "react";
import { IconChevronLeft, IconArrowNarrowUp } from "./Icons";

export default function BackToTopButton() {
  const [visible, setVisible] = useState(false);
  const progressRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    let ticking = false;

    const handleScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          const scrollTotal =
            document.documentElement.scrollHeight -
            document.documentElement.clientHeight;
          const scrollTop = document.documentElement.scrollTop;
          const scrollPercent = Math.floor((scrollTop / scrollTotal) * 100);

          if (progressRef.current) {
            progressRef.current.style.setProperty(
              "background-image",
              `conic-gradient(var(--accent), var(--accent) ${scrollPercent}%, transparent ${scrollPercent}%)`
            );
          }

          setVisible(scrollTop / scrollTotal > 0.3);
          ticking = false;
        });
        ticking = true;
      }
    };

    document.addEventListener("scroll", handleScroll);
    return () => document.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
  };

  return (
    <div
      className={`fixed end-4 bottom-8 z-50 md:sticky md:end-auto md:float-end md:me-1 transition duration-500 ${
        visible
          ? "translate-y-0 opacity-100"
          : "translate-y-14 opacity-0"
      }`}
    >
      <button
        onClick={scrollToTop}
        className="group relative bg-background px-2 py-1 size-14 rounded-full shadow-xl md:h-8 md:w-fit md:rounded-md md:shadow-none md:focus-visible:rounded-none md:bg-background/35 md:bg-clip-padding md:backdrop-blur-lg"
      >
        <span
          ref={progressRef}
          className="absolute inset-0 -z-10 block size-14 scale-110 rounded-full bg-transparent md:hidden md:h-8 md:rounded-md"
        />
        <IconChevronLeft className="inline-block rotate-90 md:hidden" />
        <span className="sr-only text-sm group-hover:text-accent md:not-sr-only">
          <IconArrowNarrowUp className="inline-block size-4" />
          Back To Top
        </span>
      </button>
    </div>
  );
}
