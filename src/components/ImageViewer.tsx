"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface ViewerImage {
  src: string;
  alt: string;
  width: number;
  height: number;
  isSvg: boolean;
}

interface Transform {
  x: number;
  y: number;
  scale: number;
}

const MIN_SCALE = 0.1;
const MAX_SCALE = 8;
const ZOOM_STEP = 1.25;
const VIEWPORT_PADDING = 32;

const clampScale = (scale: number) =>
  Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));

export default function ImageViewer() {
  const [image, setImage] = useState<ViewerImage | null>(null);
  const [transform, setTransform] = useState<Transform>({
    x: 0,
    y: 0,
    scale: 1,
  });
  const [isDragging, setIsDragging] = useState(false);

  const stageRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const lastFocusedRef = useRef<HTMLElement | null>(null);
  const pointersRef = useRef(new Map<number, { x: number; y: number }>());
  const pinchRef = useRef<{ distance: number; scale: number } | null>(null);

  const fitToStage = useCallback((target: ViewerImage) => {
    const stage = stageRef.current;
    if (!stage) return;
    const { clientWidth, clientHeight } = stage;
    const availableWidth = clientWidth - VIEWPORT_PADDING * 2;
    const availableHeight = clientHeight - VIEWPORT_PADDING * 2;
    let scale = Math.min(
      availableWidth / target.width,
      availableHeight / target.height
    );
    // Raster images get blurry when upscaled, so only SVGs grow past 100%
    if (!target.isSvg) scale = Math.min(scale, 1);
    setTransform({
      x: (clientWidth - target.width * scale) / 2,
      y: (clientHeight - target.height * scale) / 2,
      scale,
    });
  }, []);

  const zoomAt = useCallback((factor: number, px?: number, py?: number) => {
    const stage = stageRef.current;
    if (!stage) return;
    const cx = px ?? stage.clientWidth / 2;
    const cy = py ?? stage.clientHeight / 2;
    setTransform(prev => {
      const scale = clampScale(prev.scale * factor);
      const ratio = scale / prev.scale;
      return {
        x: cx - (cx - prev.x) * ratio,
        y: cy - (cy - prev.y) * ratio,
        scale,
      };
    });
  }, []);

  const close = useCallback(() => {
    setImage(null);
    pointersRef.current.clear();
    pinchRef.current = null;
    lastFocusedRef.current?.focus();
  }, []);

  // Open the viewer when an article image is clicked
  useEffect(() => {
    const article = document.getElementById("article");
    if (!article) return;

    const images = Array.from(article.querySelectorAll("img")).filter(
      img => !img.closest("a")
    );

    const open = (img: HTMLImageElement) => {
      lastFocusedRef.current = img;
      setImage({
        src: img.currentSrc || img.src,
        alt: img.alt,
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height,
        isSvg: /\.svg($|\?)/i.test(img.src),
      });
    };

    const cleanups = images.map(img => {
      img.classList.add("zoomable-image");
      img.tabIndex = 0;
      img.setAttribute("role", "button");
      img.setAttribute(
        "aria-label",
        `${img.alt ? img.alt + " " : ""}이미지 크게 보기`
      );

      const handleClick = () => open(img);
      const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open(img);
        }
      };
      img.addEventListener("click", handleClick);
      img.addEventListener("keydown", handleKeyDown);
      return () => {
        img.removeEventListener("click", handleClick);
        img.removeEventListener("keydown", handleKeyDown);
      };
    });

    return () => cleanups.forEach(cleanup => cleanup());
  }, []);

  // Fit the image, lock page scroll, and handle keyboard shortcuts while open
  useEffect(() => {
    if (!image) return;

    fitToStage(image);
    closeButtonRef.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      else if (e.key === "+" || e.key === "=") zoomAt(ZOOM_STEP);
      else if (e.key === "-") zoomAt(1 / ZOOM_STEP);
      else if (e.key === "0") fitToStage(image);
    };
    const handleResize = () => fitToStage(image);

    document.addEventListener("keydown", handleKeyDown);
    window.addEventListener("resize", handleResize);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", handleResize);
    };
  }, [image, fitToStage, zoomAt, close]);

  // Wheel zoom needs a non-passive listener to prevent page scrolling
  useEffect(() => {
    const stage = stageRef.current;
    if (!image || !stage) return;
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = stage.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.002);
      zoomAt(factor, e.clientX - rect.left, e.clientY - rect.top);
    };
    stage.addEventListener("wheel", handleWheel, { passive: false });
    return () => stage.removeEventListener("wheel", handleWheel);
  }, [image, zoomAt]);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest("button")) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointersRef.current.size === 2) {
      const [a, b] = Array.from(pointersRef.current.values());
      pinchRef.current = {
        distance: Math.hypot(a.x - b.x, a.y - b.y),
        scale: transform.scale,
      };
    }
    setIsDragging(true);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const previous = pointersRef.current.get(e.pointerId);
    if (!previous) return;
    pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (pointersRef.current.size === 2 && pinchRef.current) {
      const [a, b] = Array.from(pointersRef.current.values());
      const rect = e.currentTarget.getBoundingClientRect();
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      const targetScale = clampScale(
        (pinchRef.current.scale * distance) / pinchRef.current.distance
      );
      zoomAt(
        targetScale / transform.scale,
        (a.x + b.x) / 2 - rect.left,
        (a.y + b.y) / 2 - rect.top
      );
      return;
    }

    setTransform(prev => ({
      ...prev,
      x: prev.x + e.clientX - previous.x,
      y: prev.y + e.clientY - previous.y,
    }));
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    pointersRef.current.delete(e.pointerId);
    if (pointersRef.current.size < 2) pinchRef.current = null;
    if (pointersRef.current.size === 0) setIsDragging(false);
  };

  const handleDoubleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!image) return;
    const rect = e.currentTarget.getBoundingClientRect();
    zoomAt(2, e.clientX - rect.left, e.clientY - rect.top);
  };

  if (!image) return null;

  const percent = Math.round(transform.scale * 100);
  const buttonClass =
    "flex h-9 min-w-9 items-center justify-center rounded-md border border-white/20 bg-black/60 px-2 text-sm text-white hover:bg-black/80 focus-visible:outline-2 focus-visible:outline-white";

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={image.alt || "이미지 크게 보기"}
      className="fixed inset-0 z-50 bg-black/85"
      data-pagefind-ignore
    >
      <div
        ref={stageRef}
        className={`absolute inset-0 touch-none overflow-hidden select-none ${
          isDragging ? "cursor-grabbing" : "cursor-grab"
        }`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onDoubleClick={handleDoubleClick}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={image.src}
          alt={image.alt}
          draggable={false}
          className="absolute top-0 left-0 max-w-none origin-top-left rounded-sm bg-white"
          style={{
            width: image.width,
            height: image.height,
            transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
          }}
        />
      </div>

      <div className="absolute top-3 right-3 flex items-center gap-2">
        <button
          type="button"
          className={buttonClass}
          onClick={() => zoomAt(1 / ZOOM_STEP)}
          aria-label="축소"
        >
          −
        </button>
        <button
          type="button"
          className={`${buttonClass} w-16 tabular-nums`}
          onClick={() => fitToStage(image)}
          aria-label="화면에 맞추기"
          title="화면에 맞추기 (0)"
        >
          {percent}%
        </button>
        <button
          type="button"
          className={buttonClass}
          onClick={() => zoomAt(ZOOM_STEP)}
          aria-label="확대"
        >
          +
        </button>
        <a
          href={image.src}
          target="_blank"
          rel="noopener noreferrer"
          className={`${buttonClass} no-underline`}
          aria-label="원본 이미지를 새 탭에서 열기"
          title="원본 열기"
        >
          ↗
        </a>
        <button
          ref={closeButtonRef}
          type="button"
          className={buttonClass}
          onClick={close}
          aria-label="닫기"
          title="닫기 (Esc)"
        >
          ✕
        </button>
      </div>

      {image.alt && (
        <p className="pointer-events-none absolute inset-x-0 bottom-4 mx-auto w-fit max-w-[90vw] rounded-md bg-black/60 px-3 py-1.5 text-center text-sm text-white">
          {image.alt}
        </p>
      )}
    </div>
  );
}
