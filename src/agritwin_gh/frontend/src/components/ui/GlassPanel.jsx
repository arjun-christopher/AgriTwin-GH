/**
 * GlassPanel — reusable frosted-glass surface.
 * Applies the `.glass-panel` utility from index.css plus
 * a configurable border-radius and any extra className.
 */
function GlassPanel({ children, className = '', as: Tag = 'div', ...props }) {
  return (
    <Tag className={`glass-panel rounded-xl ${className}`} {...props}>
      {children}
    </Tag>
  );
}

export default GlassPanel;
