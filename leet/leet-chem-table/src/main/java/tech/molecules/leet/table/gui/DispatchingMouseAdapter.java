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

    private boolean dispatchMouseReleased = false;
    private boolean dispatchMousePressed  = true;
    private boolean dispatchMouseClicked  = false;
    private boolean dispatchMouseEntered  = false;
    private boolean dispatchMouseExited   = false;


    public DispatchingMouseAdapter(Supplier<Component> dispatchToComponent) {
        this.dispatchToComponent = dispatchToComponent;
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        if (dispatchMouseClicked) {
            Component dtc = this.dispatchToComponent.get();
            dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(), e, dtc));
        }
    }

    @Override
    public void mousePressed(MouseEvent e) {
        if (dispatchMousePressed) {
            Component dtc = this.dispatchToComponent.get();
            dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(), e, dtc));
        }
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        if(this.dispatchMouseReleased) {
            Component dtc = this.dispatchToComponent.get();
            dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(), e, dtc));
        }
    }

    @Override
    public void mouseEntered(MouseEvent e) {
        if (dispatchMouseEntered) {
            Component dtc = this.dispatchToComponent.get();
            dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(), e, dtc));
        }
    }

    @Override
    public void mouseExited(MouseEvent e) {
        if (dispatchMouseExited) {
            Component dtc = this.dispatchToComponent.get();
            dtc.dispatchEvent(SwingUtilities.convertMouseEvent((Component) e.getSource(), e, dtc));
        }
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
