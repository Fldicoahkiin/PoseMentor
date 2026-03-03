import { forwardRef } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';
import './Button.scss';

const buttonVariants = cva(
    'btn-base',
    {
        variants: {
            variant: {
                default: 'btn-default',
                destructive: 'btn-destructive',
                outline: 'btn-outline',
                secondary: 'btn-secondary',
                ghost: 'btn-ghost',
                link: 'btn-link',
            },
            size: {
                default: 'btn-size-default',
                sm: 'btn-size-sm',
                lg: 'btn-size-lg',
                icon: 'btn-size-icon',
            },
        },
        defaultVariants: {
            variant: 'default',
            size: 'default',
        },
    }
);

export interface ButtonProps
    extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
    asChild?: boolean;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant, size, asChild = false, ...props }, ref) => {
        return (
            <button
                className={cn(buttonVariants({ variant, size, className }))}
                ref={ref}
                {...props}
            />
        );
    }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
