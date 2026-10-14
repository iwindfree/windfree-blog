"use client";

import { useSyncExternalStore } from "react";
import Link from "next/link";
import { IconChevronLeft } from "./Icons";
import { SITE } from "@/config";

// No external source to watch; the server snapshot keeps hydration consistent
const subscribe = () => () => {};

export default function BackButton() {
  const backUrl = useSyncExternalStore(
    subscribe,
    () => sessionStorage.getItem("backUrl") || "/",
    () => "/"
  );

  if (!SITE.showBackButton) return null;

  return (
    <div className="app-layout flex items-center justify-start">
      <Link
        href={backUrl}
        className="focus-outline -ms-2 mt-8 mb-2 inline-flex items-center gap-1 hover:text-foreground/75"
      >
        <IconChevronLeft className="inline-block size-6 rtl:rotate-180" />
        <span>Go back</span>
      </Link>
    </div>
  );
}
