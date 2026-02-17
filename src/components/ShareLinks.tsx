"use client";

import { SHARE_LINKS } from "@/constants";
import {
  IconWhatsapp,
  IconFacebook,
  IconBrandX,
  IconTelegram,
  IconPinterest,
  IconMail,
} from "./Icons";

const iconMap: Record<
  string,
  React.ComponentType<React.SVGProps<SVGSVGElement>>
> = {
  WhatsApp: IconWhatsapp,
  Facebook: IconFacebook,
  X: IconBrandX,
  Telegram: IconTelegram,
  Pinterest: IconPinterest,
  Mail: IconMail,
};

export default function ShareLinks() {
  const url = typeof window !== "undefined" ? window.location.href : "";

  if (SHARE_LINKS.length === 0) return null;

  return (
    <div className="flex flex-none flex-col items-center justify-center gap-1 md:items-start">
      <span className="italic">Share this post on:</span>
      <div className="text-center">
        {SHARE_LINKS.map((social) => {
          const Icon = iconMap[social.name];
          return (
            <a
              key={social.name}
              href={`${social.href}${url}`}
              target="_blank"
              rel="noopener noreferrer"
              className="group inline-flex scale-90 items-center gap-1 p-2 hover:rotate-6 hover:text-accent sm:p-1"
              title={social.linkTitle}
            >
              {Icon && (
                <Icon className="inline-block size-6 scale-125 fill-transparent stroke-current stroke-2 opacity-90 sm:scale-110" />
              )}
              <span className="sr-only">{social.linkTitle}</span>
            </a>
          );
        })}
      </div>
    </div>
  );
}
