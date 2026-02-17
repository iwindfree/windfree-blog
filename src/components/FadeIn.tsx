"use client";

import { useEffect, useRef } from "react";

interface FadeInProps {
  children: React.ReactNode;
  className?: string;
  stagger?: boolean;
}

export default function FadeIn({
  children,
  className = "",
  stagger = false,
}: FadeInProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const target = entry.target as HTMLElement;
            target.classList.add("is-visible");

            if (stagger) {
              const children = Array.from(target.children) as HTMLElement[];
              children.forEach((child, i) => {
                child.style.transitionDelay = `${i * 80}ms`;
              });
            }

            observer.unobserve(target);
          }
        });
      },
      {
        threshold: 0.1,
        rootMargin: "0px 0px -40px 0px",
      }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [stagger]);

  return (
    <div
      ref={ref}
      className={`${stagger ? "fade-in-stagger" : "fade-in"} ${className}`}
    >
      {children}
    </div>
  );
}
