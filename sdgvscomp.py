import matplotlib.pyplot as plt
import numpy as np

approaches = ["Aligning Practices (32.6%)", "Developed Strategies (22.8%)", "Developed Methods (0.2%)"]
percentages = [32.6, 22.8, 0.2]

plt.figure(figsize=(8, 4))
plt.barh(approaches, percentages, color=['skyblue', 'lightgreen', 'coral'])
plt.xlabel("Percentage of Corporations")
plt.title("SDG Implementation Approaches: Beyond the Buzzwords")
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

sdgs = [f"SDG {i}" for i in range(1, 18)]
percentages = [30, 22, 49, 42, 43, 32, 50, 72, 50, 33, 37, 58, 63, 18, 9, 25, 31]

high_profile_sdgs = ["SDG 7", "SDG 8", "SDG 9", "SDG 12", "SDG 13"]
high_profile_indices = [sdgs.index(sdg) for sdg in high_profile_sdgs]
high_profile_values = [percentages[i] for i in high_profile_indices]

other_sdgs = [sdg for sdg in sdgs if sdg not in high_profile_sdgs]
other_indices = [sdgs.index(sdg) for sdg in other_sdgs]
other_values = [percentages[i] for i in other_indices]

plt.figure(figsize=(12, 6))
bar_width = 0.4

plt.bar(high_profile_indices, high_profile_values, bar_width, label="High-Profile/Business-Aligned SDGs", color='skyblue')
plt.bar(np.array(other_indices) + bar_width, other_values, bar_width, label="Less Popular SDGs", color='coral')

plt.xlabel("Sustainable Development Goals (SDGs)")
plt.ylabel("Percentage of Corporations Engaging")
plt.title("The Popularity Contest (2022)")
plt.xticks(range(len(sdgs)), sdgs, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()