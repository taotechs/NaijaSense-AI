export type TaskAPreset = {
  id: string;
  label: string;
  user_persona: string;
  product_details: string;
};

export const TASK_A_PRESETS: TaskAPreset[] = [
  {
    id: "lagos_foodie",
    label: "Lagos foodie · balanced",
    user_persona:
      "Mid-20s professional in VI who eats out on weekends. Honest, value-conscious, " +
      "mentions price and wait time. Balanced tone - not harsh, not hype.",
    product_details:
      "Iya Eba Amala Spot - Saturday lunch. Soft amala, rich egusi, ₦2k per person, ~20 min wait.",
  },
  {
    id: "student_critical",
    label: "Student · budget · critical",
    user_persona:
      "UNILAG student in Yaba on a tight budget. Critical when portions are small or overpriced. " +
      "Casual Nigerian English, focuses on whether it's worth the money.",
    product_details:
      "Suya stand near campus - ₦1,500 for two sticks, pepper was good but meat felt dry, 25 min queue.",
  },
];
