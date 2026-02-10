import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  isLoading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = "primary",
  isLoading,
  className = "",
  disabled,
  ...props
}) => {
  const baseStyles =
    "px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900";

  const variants = {
    primary: "bg-cyan-500/90 hover:bg-cyan-400 text-slate-950 focus:ring-cyan-400",
    secondary:
      "bg-slate-900 hover:bg-slate-800 text-slate-100 border border-slate-800 focus:ring-slate-500",
    danger: "bg-rose-500/90 hover:bg-rose-400 text-slate-950 focus:ring-rose-400",
    ghost: "bg-transparent hover:bg-slate-900 text-slate-300 border border-slate-800",
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${
        isLoading || disabled ? "opacity-50 cursor-not-allowed" : ""
      } ${className}`}
      disabled={isLoading || disabled}
      {...props}
    >
      {isLoading && (
        <svg
          className="animate-spin -ml-1 mr-2 h-4 w-4 text-current"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          ></circle>
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          ></path>
        </svg>
      )}
      {children}
    </button>
  );
};
