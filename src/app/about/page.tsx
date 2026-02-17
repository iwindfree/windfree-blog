import type { Metadata } from "next";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "About",
};

export default function AboutPage() {
  return (
    <>
      <Header />

      {/* Full-width hero with gradient overlay */}
      <div className="relative">
        <img
          src="/images/about/about-hero.jpeg"
          alt="About"
          className="max-h-[560px] w-full object-cover object-[center_70%]"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" />
        <div className="absolute bottom-8 left-0 right-0 text-center">
          <h1 className="text-3xl font-bold text-white drop-shadow-lg sm:text-4xl">
            About
          </h1>
          <p className="mt-2 text-sm text-white/80 drop-shadow">
            Notes for a Some day
          </p>
        </div>
      </div>

      <main id="main-content" className="mx-auto w-full max-w-app px-4 py-12">
        <section
          id="about"
          className="app-prose mb-28 max-w-app [word-break:keep-all] prose-img:border-0 prose-img:rounded-xl prose-img:shadow-[0_2px_16px_rgba(0,0,0,0.08)]"
        >
          <div className="mx-auto mt-4 mb-12 flex flex-col items-center text-center">
            <img
              src="/images/windfree-logo.jpeg"
              alt="windfree"
              className="!mb-4 !mt-0 h-32 w-32 !rounded-full !border-0 object-cover !shadow-none"
            />
            <h2 className="!mb-1 !mt-0 text-2xl">iwindfree</h2>
            <p className="!mt-0 text-sm text-accent-gold">
              computer programmer
            </p>
          </div>

          <p>
            개발자로 참 오랜 시간을 보낸 것 같습니다. 대학 시절 아무것도 모르는
            상태에서 프로그래밍에 흥미를 느껴 이 분야에서 일을 시작한 것이 벌써
            20년을 훌쩍 넘게 되었네요. 운 좋게도 하고 싶은 것들을 많이 하면서
            보낼 수 있었습니다. 좋은 동료들과 계속 일을 할 수 있었고, 만들고 싶은
            솔루션도 계속 만들 수 있었으니까요.
          </p>

          <p>
            워낙 기술이 빨리 변해서 요새는 따라가기에도 힘이 드는 것이 사실이지만,
            그 빠른 변화의 흐름을 지켜보는 것도 무척 흥미롭습니다.
          </p>

          <p>
            어느 순간, 이것 저것 많이 해보았는데 점점 기억에서 많이 지워지는 것
            같더군요. 많이 늦은 감이 있지만 지금부터라도 새롭게 공부하거나 도움이
            될 만한 내용을 기록으로 남겨 보려 합니다.
          </p>

          <hr />

          <h2>추억</h2>

          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <figure>
              <img src="/images/about/myhome.jpg" alt="어릴적 살던 초가집" />
              <figcaption>
                <em>어릴적 살던 초가집, 일출봉은 배경일 뿐…</em>
              </figcaption>
            </figure>

            <figure>
              <img src="/images/about/myhome2.jpg" alt="고2때부터는 여기에서" />
              <figcaption>
                <em>고2때부터는 여기에서..</em>
              </figcaption>
            </figure>

            <figure>
              <img src="/images/about/marathon.jpg" alt="2010년 춘마" />
              <figcaption>
                <em>2010년 춘마</em>
              </figcaption>
            </figure>

            <figure>
              <img src="/images/about/guitar.jpg" alt="제일 좋아하는" />
              <figcaption>
                <em>제일 좋아하는…</em>
              </figcaption>
            </figure>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}
