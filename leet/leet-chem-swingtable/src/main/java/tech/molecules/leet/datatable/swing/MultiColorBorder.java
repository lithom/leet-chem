package tech.molecules.leet.datatable.swing;

import javax.swing.border.AbstractBorder;
import java.awt.*;
import java.util.List;


public class MultiColorBorder extends AbstractBorder {
    private final List<Color> colors;
    private final int stretchLength;

    public MultiColorBorder(List<Color> colors, int stretchLength) {
        this.colors = colors;
        this.stretchLength = stretchLength;
    }

    @Override
    public void paintBorder(Component c, Graphics g, int x, int y, int width, int height) {
        Graphics2D g2d = (Graphics2D) g.create();
        g2d.setStroke(new BasicStroke(1));

        int[] xPoints = {x, x + width, x + width, x, x};
        int[] yPoints = {y, y, y + height, y + height, y};

        for (int i = 0; i < xPoints.length - 1; i++) {
            int startX = xPoints[i];
            int startY = yPoints[i];
            int endX = xPoints[i + 1];
            int endY = yPoints[i + 1];

            int dx = endX - startX;
            int dy = endY - startY;
            int distance = (int) Math.sqrt(dx * dx + dy * dy);

            for (int j = 0; j < distance; j += stretchLength) {
                Color color = colors.get((j / stretchLength) % colors.size());
                g2d.setColor(color);
                g2d.drawLine(
                        startX + (j * dx) / distance,
                        startY + (j * dy) / distance,
                        startX + (Math.min(j + stretchLength, distance) * dx) / distance,
                        startY + (Math.min(j + stretchLength, distance) * dy) / distance
                );
            }
        }

        g2d.dispose();
    }

    @Override
    public Insets getBorderInsets(Component c) {
        int borderWidth = 1;
        return new Insets(borderWidth, borderWidth, borderWidth, borderWidth);
    }
}