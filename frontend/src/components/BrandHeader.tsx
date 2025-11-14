import clsx from "clsx";

type BrandHeaderProps = {
  isVisible: boolean;
};

export const BrandHeader = ({ isVisible }: BrandHeaderProps) => (
  <header className={clsx("brand-header", isVisible && "visible")}>
    <div className="brand-lockup">
      <span className="brand-mark">Praedium</span>
      <span className="brand-tagline">Credit Intelligence</span>
    </div>
  </header>
);

