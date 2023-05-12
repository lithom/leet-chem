package tech.molecules.leet.datatable.swing;

import javafx.scene.control.Labeled;

import javax.swing.*;
import javax.swing.table.TableCellRenderer;
import java.awt.*;

public abstract class GridOfColoredStringsRenderer implements TableCellRenderer {

    public static class ColoredString {
        public final String str;
        public final Color col;
        public ColoredString(String str, Color col) {
            this.str = str;
            this.col = col;
        }
    }

    public static class GridOfColoredStrings {
        public final int rows;
        public final int cols;
        public final ColoredString[] cells;
        public GridOfColoredStrings(int rows, int cols, ColoredString[] cells) {
            this.rows = rows;
            this.cols = cols;
            this.cells = cells;
        }
    }

    public abstract GridOfColoredStrings convertToGridOfColoredStrings(Object obj);

    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int col) {

        GridOfColoredStrings grid = convertToGridOfColoredStrings(value) ;
        if(grid==null) {
            return new JPanel();
        }

        JPanel panel = new JPanel(new GridLayout(grid.rows, grid.cols));

        for (ColoredString coloredString : grid.cells) {
            JLabel label = new JLabel(coloredString.str);
            label.setBackground(coloredString.col);
            label.setOpaque(true); // This is necessary to see the background color.
            panel.add(label);
        }

        return panel;
    }

}
