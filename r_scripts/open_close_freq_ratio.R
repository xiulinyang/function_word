library(tidyverse)

df_fw <- read_tsv(
  "/Users/xiulinyang/Desktop/TODO/function_word/ud_stats/close_vs_open_count.tsv",
  show_col_types = FALSE
)

library(tidyverse)

df_long <- df_fw %>%
  transmute(
    language,
    closed_type  = closed_ratio,
    closed_freq  = closed_freq_ratio,
    open_type    = open_ratio,
    open_freq    = open_freq_ratio
  ) %>%
  pivot_longer(
    cols = -language,
    names_to = c("class", ".value"),
    names_pattern = "(closed|open)_(type|freq)"
  ) %>%
  mutate(
    class = recode(
      class,
      "closed" = "Function words (closed class)",
      "open"   = "Content words (open class)"
    )
  )

ggplot(df_long, aes(
  x = type,
  y = freq,
  color = class
)) +
  geom_point(
    size = 2,
    alpha = 0.75
  ) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "gray50",
    linewidth = 0.7
  ) +
  scale_color_manual(
    values = c(
      "Function words (closed class)" = "#6D8CD1",
      "Content words (open class)"    = "#B64C4C"
    )
  ) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_fixed() +
  labs(
    x = "Type Ratio",
    y = "Token Frequency Ratio",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    legend.position = c(0.97, 0.03),
    legend.justification = c("right", "bottom"),
    # legend.background = element_rect(
    #   fill = "white",
    #   color = "black",
    #   linewidth = 0.4
    # ),
    legend.key = element_blank(),
    legend.text = element_text(size = 11),

    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18)
  )