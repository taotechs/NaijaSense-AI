/** Preset comprehensive personas for the Task B demo (hackathon: persona-only input). */

export type PersonaPreset = {
  id: string;
  label: string;
  user_id: string;
  persona: string;
};

export const TASK_B_PERSONA_PRESETS: PersonaPreset[] = [
  {
    id: "student_yaba",
    label: "Student · Yaba · budget",
    user_id: "demo_student",
    persona:
      "I'm a 22-year-old UNILAG student in Yaba with about ₦10k per week for fun and food. " +
      "I want cheap jollof and suya spots, low-cost weekend Nollywood with friends, and occasional smoothies - " +
      "never premium Island prices.",
  },
  {
    id: "vi_professional",
    label: "Professional · VI · balanced",
    user_id: "demo_professional",
    persona:
      "Early-career product manager on Victoria Island. I dine out twice a week, enjoy polished seafood spots, " +
      "Afrobeats lounges, specialty coffee, and reliable tech accessories. I pay for quality but still compare value.",
  },
  {
    id: "abuja_family",
    label: "Family · Abuja · food & movies",
    user_id: "demo_family",
    persona:
      "Abuja-based parent planning family weekends: kid-friendly restaurants in Wuse, mild Nollywood picks, " +
      "and healthy drinks. Moderate budget, safety and wait times matter more than hype.",
  },
];
