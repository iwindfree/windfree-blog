import Link from "next/link";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

export default function NotFound() {
  return (
    <>
      <Header />
      <main
        id="main-content"
        className="app-layout flex flex-1 items-center justify-center"
      >
        <div className="mb-14 flex flex-col items-center justify-center">
          <h1 className="text-9xl font-bold text-accent">404</h1>
          <span aria-hidden="true">¯\_(ツ)_/¯</span>
          <p className="mt-4 text-2xl sm:text-3xl">
            페이지를 찾을 수 없습니다
          </p>
          <Link
            href="/"
            className="my-6 text-lg underline decoration-dashed underline-offset-8 hover:text-accent"
          >
            홈으로 돌아가기
          </Link>
        </div>
      </main>
      <Footer />
    </>
  );
}
