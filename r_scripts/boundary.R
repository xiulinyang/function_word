library(tidyverse)
library(ggrepel)

df_closed <- read_tsv(
  "/Users/xiulinyang/Desktop/TODO/function_word/ud_stats/boundary_stats_fixed.tsv",
  show_col_types = FALSE
) %>%
  mutate(
    Boundary_Rate = as.numeric(Boundary_Rate),
    Type = "Closed"
  ) %>%
  drop_na(Boundary_Rate)

df_open <- read_tsv(
  "/Users/xiulinyang/Desktop/TODO/function_word/ud_stats/boundary_stats_open.tsv",
  show_col_types = FALSE
) %>%
  mutate(
    Boundary_Rate = as.numeric(Boundary_Rate),
    Type = "Open"
  ) %>%
  drop_na(Boundary_Rate)

df_all <- bind_rows(df_closed, df_open)

# --- outliers ---
low_closed <- df_closed %>%
  filter(Boundary_Rate < 0.0)

high_open <- df_open %>%
  filter(Boundary_Rate > 1.0)

p <- ggplot(df_all, aes(x = Type, y = Boundary_Rate, fill = Type)) +

  ## violin
  geom_violin(
    alpha = 0.35,
    width = 0.9,
    trim = FALSE
  ) +

  ## jitter
  geom_jitter(
    width = 0.12,
    alpha = 0.25,
    size = 1.2,
    color = "gray35"
  ) +

  ## Closed: low outliers
  geom_point(
    data = low_closed,
    aes(x = "Closed", y = Boundary_Rate),
    color = "#B64C4C",
    size = 2.2,
    alpha = 0.9,
    inherit.aes = FALSE
  ) +

  geom_text_repel(
    data = low_closed,
    aes(x = "Closed", y = Boundary_Rate, label = Language),
    size = 3,
    min.segment.length = 0,
    box.padding = 0.35,
    point.padding = 0.25,
    max.overlaps = Inf,
    inherit.aes = FALSE
  ) +

  ## Open: high outliers
  geom_point(
    data = high_open,
    aes(x = "Open", y = Boundary_Rate),
    color = "#B64C4C",
    size = 2.2,
    alpha = 0.9,
    inherit.aes = FALSE
  ) +

  geom_text_repel(
    data = high_open,
    aes(x = "Open", y = Boundary_Rate, label = Language),
    size = 3,
    min.segment.length = 0,
    box.padding = 0.35,
    point.padding = 0.25,
    max.overlaps = Inf,
    inherit.aes = FALSE
  ) +

  ## reference lines
  geom_hline(yintercept = 0.8, linetype = "dotted", color = "gray50") +

  coord_flip() +

  scale_fill_manual(
    values = c(
      "Closed" = "#6D8CD1",
      "Open"   = "#C74A4A"
    )
  ) +

  labs(
    x = NULL,
    y = "Boundary Rate",
    fill = NULL
  ) +

  theme_minimal(base_size = 15) +
  theme(
    panel.grid.minor = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "none",
    axis.text.y = element_text(
      angle = 90,
      size  = 15
    )
  )

p