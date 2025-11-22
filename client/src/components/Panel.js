const Panel = ({ className = "", children }) => (
  <div
    className={`flex-1 bg-white rounded-[28px] shadow-[0_20px_45px_rgba(15,23,42,0.08)] p-8 ${className}`}
  >
    {children}
  </div>
);

export default Panel;
