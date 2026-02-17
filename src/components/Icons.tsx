import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const defaultProps: IconProps = {
  width: 24,
  height: 24,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export function IconX(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M18 6l-12 12" />
      <path d="M6 6l12 12" />
    </svg>
  );
}

export function IconMoon(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 0 0 7.92 12.446a9 9 0 1 1 -8.313 -12.454z" />
    </svg>
  );
}

export function IconSunHigh(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M14.828 14.828a4 4 0 1 0 -5.656 -5.656a4 4 0 0 0 5.656 5.656z" />
      <path d="M6.343 17.657l-1.414 1.414" />
      <path d="M6.343 6.343l-1.414 -1.414" />
      <path d="M17.657 6.343l1.414 -1.414" />
      <path d="M17.657 17.657l1.414 1.414" />
      <path d="M4 12h-2" />
      <path d="M12 4v-2" />
      <path d="M20 12h2" />
      <path d="M12 20v2" />
    </svg>
  );
}

export function IconSearch(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M10 10m-7 0a7 7 0 1 0 14 0a7 7 0 1 0 -14 0" />
      <path d="M21 21l-6 -6" />
    </svg>
  );
}

export function IconMenuDeep(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M4 6h16" />
      <path d="M7 12h13" />
      <path d="M10 18h10" />
    </svg>
  );
}

export function IconArrowRight(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M5 12l14 0" />
      <path d="M13 18l6 -6" />
      <path d="M13 6l6 6" />
    </svg>
  );
}

export function IconArrowLeft(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M5 12l14 0" />
      <path d="M5 12l6 6" />
      <path d="M5 12l6 -6" />
    </svg>
  );
}

export function IconArrowNarrowUp(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M12 5l0 14" />
      <path d="M16 9l-4 -4" />
      <path d="M8 9l4 -4" />
    </svg>
  );
}

export function IconCalendar(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M4 7a2 2 0 0 1 2 -2h12a2 2 0 0 1 2 2v12a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2v-12z" />
      <path d="M16 3v4" />
      <path d="M8 3v4" />
      <path d="M4 11h16" />
      <path d="M7 14h.013" />
      <path d="M10.01 14h.005" />
      <path d="M13.01 14h.005" />
      <path d="M16.015 14h.005" />
      <path d="M13.015 17h.005" />
      <path d="M7.01 17h.005" />
      <path d="M10.01 17h.005" />
    </svg>
  );
}

export function IconHash(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M5 9l14 0" />
      <path d="M5 15l14 0" />
      <path d="M11 4l-4 16" />
      <path d="M17 4l-4 16" />
    </svg>
  );
}

export function IconChevronLeft(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M15 6l-6 6l6 6" />
    </svg>
  );
}

export function IconChevronRight(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M9 6l6 6l-6 6" />
    </svg>
  );
}

export function IconGitHub(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M9 19c-4.3 1.4 -4.3 -2.5 -6 -3m12 5v-3.5c0 -1 .1 -1.4 -.5 -2c2.8 -.3 5.5 -1.4 5.5 -6a4.6 4.6 0 0 0 -1.3 -3.2a4.2 4.2 0 0 0 -.1 -3.2s-1.1 -.3 -3.5 1.3a12.3 12.3 0 0 0 -6.2 0c-2.4 -1.6 -3.5 -1.3 -3.5 -1.3a4.2 4.2 0 0 0 -.1 3.2a4.6 4.6 0 0 0 -1.3 3.2c0 4.6 2.7 5.7 5.5 6c-.6 .6 -.6 1.2 -.5 2v3.5" />
    </svg>
  );
}

export function IconEdit(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M7 7h-1a2 2 0 0 0 -2 2v9a2 2 0 0 0 2 2h9a2 2 0 0 0 2 -2v-1" />
      <path d="M20.385 6.585a2.1 2.1 0 0 0 -2.97 -2.97l-8.415 8.385v3h3l8.385 -8.415z" />
      <path d="M16 5l3 3" />
    </svg>
  );
}

export function IconWhatsapp(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M3 21l1.65 -3.8a9 9 0 1 1 3.4 2.9l-5.05 .9" />
      <path d="M9 10a.5 .5 0 0 0 1 0v-1a.5 .5 0 0 0 -1 0v1a5 5 0 0 0 5 5h1a.5 .5 0 0 0 0 -1h-1a.5 .5 0 0 0 0 1" />
    </svg>
  );
}

export function IconFacebook(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M7 10v4h3v7h4v-7h3l1 -4h-4v-2a1 1 0 0 1 1 -1h3v-4h-3a5 5 0 0 0 -5 5v2h-3" />
    </svg>
  );
}

export function IconBrandX(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M4 4l11.733 16h4.267l-11.733 -16z" />
      <path d="M4 20l6.768 -6.768m2.46 -2.46l6.772 -6.772" />
    </svg>
  );
}

export function IconTelegram(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M15 10l-4 4l6 6l4 -16l-18 7l4 2l2 6l3 -4" />
    </svg>
  );
}

export function IconPinterest(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M8 20l4 -9" />
      <path d="M10.7 14c.437 1.263 1.43 2 2.55 2c2.071 0 3.75 -1.554 3.75 -4a5 5 0 1 0 -9.7 1.7" />
      <path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" />
    </svg>
  );
}

export function IconMail(props: IconProps) {
  return (
    <svg {...defaultProps} {...props}>
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M3 7a2 2 0 0 1 2 -2h14a2 2 0 0 1 2 2v10a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-10z" />
      <path d="M3 7l9 6l9 -6" />
    </svg>
  );
}
