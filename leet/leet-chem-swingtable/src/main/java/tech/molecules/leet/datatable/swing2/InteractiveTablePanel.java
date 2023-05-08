package tech.molecules.leet.datatable.swing2;

import javax.swing.*;
import java.awt.*;

public class InteractiveTablePanel extends JPanel implements Scrollable {

    private InteractiveTableModel model;

    public InteractiveTablePanel(InteractiveTableModel model) {
        this.setDoubleBuffered(true);
        this.model = model;
        setLayout(null); // Use a null (absolute) layout
        reinitLayout();
    }

    public void setModel(InteractiveTableModel model) {
        this.model = model;
        reinitLayout();
    }

    public void reinitLayout() {
        removeAll(); // Remove all existing components

        for( int row = 0; row < model.getRows() ; row++) {

            int rowY = model.getRow_Y(row);
            int hY = model.getRow_H(row);

            for(int col = 0; col < model.getColumns() ; col++) {
                int colX = model.getCol_X(col);
                int wX = model.getCol_W(col);
                JComponent component = model.createComponent(row, col);
                if (component != null) {
                    component.setLocation(colX, rowY);
                    component.setSize(wX,hY);
                    add(component);
                }
            }
        }

        setPreferredSize( new Dimension(model.getCol_X(model.getColumns()+1) , model.getRow_Y(model.getRows()+1)) );
        revalidate();
        repaint();
    }


    @Override
    public Dimension getPreferredScrollableViewportSize() {
        return getPreferredSize();
    }

    @Override
    public int getScrollableUnitIncrement(Rectangle visibleRect, int orientation, int direction) {
        return 16;
    }

    @Override
    public int getScrollableBlockIncrement(Rectangle visibleRect, int orientation, int direction) {
        return 16;
    }

    @Override
    public boolean getScrollableTracksViewportWidth() {
        return false;
    }

    @Override
    public boolean getScrollableTracksViewportHeight() {
        return false;
    }
}
