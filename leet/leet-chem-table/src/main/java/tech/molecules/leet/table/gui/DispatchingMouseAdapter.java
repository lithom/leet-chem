package tech.molecules.leet.table.gui;

import javax.swing.*;
import javax.swing.plaf.basic.BasicTreeUI;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.function.Supplier;

public class DispatchingMouseAdapter implements MouseListener, MouseMotionListener {

    private Supplier<Component> dispatchToComponent;

    public DispatchingMouseAdapter(Supplier<Component> dispatchToComponent) {
        this.dispatchToComponent = dispatchToComponent;
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        Component dtc = this.dispatchToComponent.get();
        dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(),e,dtc));
    }

    @Override
    public void mousePressed(MouseEvent e) {
        Component dtc = this.dispatchToComponent.get();
        dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(),e,dtc));
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        Component dtc = this.dispatchToComponent.get();
        dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(),e,dtc));
    }

    @Override
    public void mouseEntered(MouseEvent e) {
        Component dtc = this.dispatchToComponent.get();
        dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(),e,dtc));
    }

    @Override
    public void mouseExited(MouseEvent e) {
        Component dtc = this.dispatchToComponent.get();
        dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(),e,dtc));
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        Component dtc = this.dispatchToComponent.get();
        dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(),e,dtc));
    }

    @Override
    public void mouseMoved(MouseEvent e) {
        Component dtc = this.dispatchToComponent.get();
        dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(),e,dtc));
    }
}
