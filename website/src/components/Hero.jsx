export default function Hero() {
  return (
    <section className="hero" id="home">
      <div className="hero-blob blob1" />
      <div className="hero-blob blob2" />
      <h1 className="hero-title">
        guiding firefighters<br /><em>back to safety</em>
      </h1>
      <p className="hero-sub">
        A two-way visual heads up display that provides firefighters real-time navigation.
      </p>
      <div className="hero-actions">
        <a href="#solution" className="btn-primary">See the Solution</a>
        <a href="#product" className="btn-ghost">View Product</a>
      </div>
    </section>
  );
}
