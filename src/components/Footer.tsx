import Socials from "./Socials";

interface FooterProps {
  noMarginTop?: boolean;
  className?: string;
}

export default function Footer({
  noMarginTop = false,
  className = "",
}: FooterProps) {
  const currentYear = new Date().getFullYear();

  return (
    <footer
      className={`app-layout ${!noMarginTop ? "mt-auto" : ""} ${className}`}
    >
      <div className="flex flex-col items-center justify-between border-t border-border py-6 sm:flex-row-reverse sm:py-4">
        <Socials />
        <div className="my-2 flex flex-col items-center whitespace-nowrap sm:flex-row">
          <span className="font-semibold text-accent">
            WINDFREE&apos;S BLOG
          </span>
          <span className="hidden sm:inline">&nbsp;|&nbsp;</span>
          <span>Copyright &copy; {currentYear} All rights reserved.</span>
        </div>
      </div>
    </footer>
  );
}
