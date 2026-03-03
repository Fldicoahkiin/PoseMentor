import { clsx, type ClassValue } from "clsx"
import { extendTailwindMerge } from "tailwind-merge"

// We create a custom tailwind merge instance to support standard custom CSS classes if any
const customTwMerge = extendTailwindMerge({
    extend: {
        // Add custom class definitions here if needed in the future
    },
})

export function cn(...inputs: ClassValue[]) {
    return customTwMerge(clsx(inputs))
}
