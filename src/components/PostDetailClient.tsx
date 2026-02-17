"use client";

import { useEffect } from "react";

export default function PostDetailClient() {
  useEffect(() => {
    // Progress bar
    const progressContainer = document.createElement("div");
    progressContainer.className =
      "progress-container fixed top-0 z-10 h-1 w-full bg-background";
    const progressBar = document.createElement("div");
    progressBar.className = "progress-bar h-1 w-0 bg-accent";
    progressBar.id = "myBar";
    progressContainer.appendChild(progressBar);
    document.body.appendChild(progressContainer);

    const handleScroll = () => {
      const winScroll =
        document.body.scrollTop || document.documentElement.scrollTop;
      const height =
        document.documentElement.scrollHeight -
        document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      const myBar = document.getElementById("myBar");
      if (myBar) {
        myBar.style.width = scrolled + "%";
      }
    };

    document.addEventListener("scroll", handleScroll);

    // Heading links
    const headings = Array.from(
      document.querySelectorAll("h2, h3, h4, h5, h6")
    );
    for (const heading of headings) {
      heading.classList.add("group");
      const link = document.createElement("a");
      link.className =
        "heading-link ms-2 no-underline opacity-75 md:opacity-0 md:group-hover:opacity-100 md:focus:opacity-100";
      link.href = "#" + heading.id;
      const span = document.createElement("span");
      span.ariaHidden = "true";
      span.innerText = "#";
      link.appendChild(span);
      heading.appendChild(link);
    }

    // Copy buttons
    const codeBlocks = Array.from(document.querySelectorAll("pre"));
    for (const codeBlock of codeBlocks) {
      const wrapper = document.createElement("div");
      wrapper.style.position = "relative";

      const computedStyle = getComputedStyle(codeBlock);
      const hasFileNameOffset =
        computedStyle.getPropertyValue("--file-name-offset").trim() !== "";
      const topClass = hasFileNameOffset
        ? "top-(--file-name-offset)"
        : "-top-3";

      const copyButton = document.createElement("button");
      copyButton.className = `copy-code absolute end-3 ${topClass} rounded bg-muted border border-muted px-2 py-1 text-xs leading-4 text-foreground font-medium`;
      copyButton.innerHTML = "Copy";
      codeBlock.setAttribute("tabindex", "0");
      codeBlock.appendChild(copyButton);

      codeBlock?.parentNode?.insertBefore(wrapper, codeBlock);
      wrapper.appendChild(codeBlock);

      copyButton.addEventListener("click", async () => {
        const code = codeBlock.querySelector("code");
        const text = code?.innerText;
        await navigator.clipboard.writeText(text ?? "");
        copyButton.innerText = "Copied";
        setTimeout(() => {
          copyButton.innerText = "Copy";
        }, 700);
      });
    }

    return () => {
      document.removeEventListener("scroll", handleScroll);
      progressContainer.remove();
    };
  }, []);

  return null;
}
