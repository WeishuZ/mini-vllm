import type { ComponentType } from "react";

export interface ChapterStepProps {
  step: number; // 0..(steps.length - 1)
}

export interface StepDef {
  title: string;
  note?: string;
}

export interface ChapterDef {
  id: string;
  title: string;
  /**
   * Per-step planning metadata. **Length === total steps in this chapter.**
   * The React chapter component is still the source of visual implementation.
   */
  steps: StepDef[];
  Component: ComponentType<ChapterStepProps>;
}
