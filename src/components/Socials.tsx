import { SOCIALS } from "@/constants";
import { IconGitHub } from "./Icons";

const iconMap: Record<string, React.ComponentType<React.SVGProps<SVGSVGElement>>> = {
  GitHub: IconGitHub,
};

export default function Socials() {
  return (
    <div className="flex flex-wrap items-center gap-1">
      {SOCIALS.map((social) => {
        const Icon = iconMap[social.name];
        return (
          <a
            key={social.name}
            href={social.href}
            target="_blank"
            rel="noopener noreferrer"
            className="group inline-flex items-center gap-1 p-2 hover:rotate-6 hover:text-accent sm:p-1"
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
  );
}
