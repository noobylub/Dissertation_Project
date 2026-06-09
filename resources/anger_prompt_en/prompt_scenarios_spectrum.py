OPTION_A_INDIVIDUALISTIC = "Option A: individualistic norm - prioritize personal autonomy, explicit self-expression, direct boundary-setting, and defending individual rights even if it creates interpersonal tension."
OPTION_B_COLLECTIVIST = "Option B: collectivist norm - prioritize group harmony, relational obligations, role respect, and indirect conflict management to preserve social cohesion even if personal preferences are restrained."


prompts_en_spectrum = [
    f"""
You are in a long checkout line at a store, and someone cuts in front of you without asking or acknowledging you.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
During a team meeting, your coworker interrupts you repeatedly and mocks your ideas in front of everyone.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
You worked hard on a project, but your teammate takes full credit for the results during a public presentation.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
A close friend borrows money from you, promises to pay you back quickly, and then keeps avoiding the topic for months.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
Your manager blames you for a mistake that was not yours in front of the whole team.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
Someone damages your car in a parking lot and leaves without a note, and you discover it right before an important event.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
A neighbor keeps playing loud music late at night even after you have politely asked them multiple times to stop.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
You arrive on time for a medical appointment, but people who arrived later are seen before you without explanation.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
A family member shares your private information with others after you explicitly asked them not to.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
While driving, another driver nearly causes an accident, then shouts at you as if it was your fault.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
You discover someone posted an unfair and false accusation about you on social media.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
After waiting a long time at customer service, the representative speaks rudely to you and dismisses your problem.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
]
