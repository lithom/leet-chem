package tech.molecules.leet.table.gui;

import com.actelion.research.gui.VerticalFlowLayout;
import tech.molecules.leet.table.NColumn;


import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JFilterPanel extends JPanel {

    JScrollPane jsp_main = new JScrollPane();
    JPanel jp_filters    = new JPanel();

    boolean minimizable  = true;
    boolean minimized    = true;


    List<NColumn.NexusRowFilter> filters = new ArrayList<>();
    //Map<NColumn.NexusRowFilter,JPanel> filter_panels = new HashMap<>();

    public JFilterPanel() {
        this.setLayout(new BorderLayout());
        jsp_main.setViewportView(jp_filters);
        jp_filters.setLayout(new VerticalFlowLayout());
        this.add(jsp_main);

        this.setOpaque(true);
        this.setBorder(new LineBorder(Color.BLACK,4));
        this.setBackground(Color.red);

        this.addMouseListener(new MouseAdapter(){
            @Override
            public void mouseClicked(MouseEvent e) {
                if(isMinimized()) {
                    setMinimized(false);
                }
            }
        });
    }

    public boolean isMinimized() {return this.minimized;}
    public void setMinimized(boolean minim) {
        this.minimized = minim;
        this.invalidate();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if(this.minimized) {
            g.setColor(Color.blue);
            g.fillRect(0,0,16,16);
        }
    }

    @Override
    public Dimension getPreferredSize() {
        if(this.minimized) {
            return new Dimension(16,16);
        }
        else {
            return super.getPreferredSize();
        }
    }

    @Override
    public Dimension getMaximumSize() {
        return this.getPreferredSize();
    }

    @Override
    public Dimension getMinimumSize() {
        return this.getPreferredSize();
    }

    public void addFilter(NColumn.NexusRowFilter filter) {
        filters.add(filter);
        JPanel pi = filter.getFilterGUI();
        //jp_filters.add(pi);
        this.jp_filters.add(pi);
        this.jp_filters.invalidate();
        //this.jsp_main.invalidate();
        //this.invalidate();
    }

    public void removeFilter(NColumn.NexusRowFilter filter) {
        JPanel pf = this.jp_filters;//this.filter_panels.get(filter);
        if(pf!=null) {
            this.remove(pf);
            //filter_panels.remove(filter);
            filters.remove(filter);
        }
    }

}
