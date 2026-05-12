export default function Hero() {
  return (
    <section className="hero" id="home" style={{ backgroundImage: `url(${import.meta.env.BASE_URL}photos/firefighter.png)`, backgroundSize: 'cover', backgroundPosition: 'center top' }}>
      <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(10,10,10,0.92) 40%, rgba(10,10,10,0.55) 100%)', zIndex: 0 }} />
      <div className="hero-blob blob1" />
      <div className="hero-blob blob2" />
      <h1 className="hero-title" style={{ position: 'relative', zIndex: 1 }}>
        guiding firefighters<br /><em>back to safety</em>
      </h1>
      <p className="hero-sub" style={{ position: 'relative', zIndex: 1 }}>
        A two-way visual heads up display that provides firefighters real-time navigation.
      </p>
      <div className="hero-actions" style={{ position: 'relative', zIndex: 1 }}>
        <a href="#solution" className="btn-primary">See the Solution</a>
        <a href="#product" className="btn-ghost">View Product</a>
      </div>
    </section>
  );
}
