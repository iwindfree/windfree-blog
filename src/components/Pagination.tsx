import Link from "next/link";
import { IconArrowLeft, IconArrowRight } from "./Icons";

interface PaginationProps {
  currentPage: number;
  lastPage: number;
  prevUrl?: string;
  nextUrl?: string;
}

export default function Pagination({
  currentPage,
  lastPage,
  prevUrl,
  nextUrl,
}: PaginationProps) {
  if (lastPage <= 1) return null;

  return (
    <nav
      className="mt-auto mb-8 flex justify-center"
      role="navigation"
      aria-label="Pagination Navigation"
    >
      {prevUrl ? (
        <Link
          href={prevUrl}
          className="group me-4 inline-flex select-none items-center gap-1 hover:text-accent"
          aria-label="Goto Previous Page"
        >
          <IconArrowLeft className="inline-block rtl:rotate-180" />
          Prev
        </Link>
      ) : (
        <span className="me-4 inline-flex select-none items-center gap-1 opacity-50">
          <IconArrowLeft className="inline-block rtl:rotate-180" />
          Prev
        </span>
      )}
      {currentPage} / {lastPage}
      {nextUrl ? (
        <Link
          href={nextUrl}
          className="group ms-4 inline-flex select-none items-center gap-1 hover:text-accent"
          aria-label="Goto Next Page"
        >
          Next
          <IconArrowRight className="inline-block rtl:rotate-180" />
        </Link>
      ) : (
        <span className="ms-4 inline-flex select-none items-center gap-1 opacity-50">
          Next
          <IconArrowRight className="inline-block rtl:rotate-180" />
        </span>
      )}
    </nav>
  );
}
