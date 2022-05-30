package tech.molecules.leet.table.gui;

import com.actelion.research.gui.VerticalFlowLayout;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

public class JExtendedSlider extends JPanel {

    JLabel jTitle;
    JPanel            jLowerPanel;
    JSlider           jSlider;
    JTextField        jValA;
    JTextField        jValB;

    double vA;
    double vStep;

    public JExtendedSlider(String title, double range[], int steps) {
        init(title,range,steps);
    }

    public JSlider getSlider() {
        return this.jSlider;
    }

    public double getSliderValue() {
        return this.vA + this.vStep * this.jSlider.getValue();
    }

    private void init(String title, double range[], int steps) {
        this.vA = range[0];
        this.vStep = (range[1]-range[0]) / steps;

        this.setLayout(new VerticalFlowLayout());
        this.jTitle = new JLabel(title);
        this.add(jTitle);

        jLowerPanel = new JPanel();
        jLowerPanel.setLayout(new BorderLayout());
        jValA = new JTextField(4);
       //jValB = new JTextField(4);
        jValA.setEditable(false);
        //jValB.setEditable(false);
        jSlider = new JSlider(0,steps);
        this.add(jSlider);
        this.add(jLowerPanel);
        jLowerPanel.add(jValA,BorderLayout.WEST);
        //jLowerPanel.add(jValB,BorderLayout.EAST);
        this.jValA.setText(String.format("%.2f",range[0]));
        //this.jValB.setText(String.format("%.2f",range[1]));

        this.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {

            }

            @Override
            public void mousePressed(MouseEvent e) {
                maybeShowPopupMenu(e);
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                maybeShowPopupMenu(e);
            }

            @Override
            public void mouseEntered(MouseEvent e) {

            }

            @Override
            public void mouseExited(MouseEvent e) {

            }

            private void maybeShowPopupMenu(MouseEvent e) {
                if(e.isPopupTrigger()) {
                    JPopupMenu popup = new JPopupMenu();
                    JMenuItem  setRange = new JMenuItem("Set Range..");
                    popup.add(setRange);
                    setRange.addActionListener(new ActionListener(){
                        @Override
                        public void actionPerformed(ActionEvent e) {
                            // TODO
                        }
                    });
                }
            }
        });

        jSlider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                String sfa = String.format( "%.2f" , vA + vStep * jSlider.getValue()   );
                //String sfb = String.format( "%.2f" ,jSlider.getRange()[1] );
                jValA.setText(sfa);
                //jValB.setText(sfb);
            }
        });
    }

}
