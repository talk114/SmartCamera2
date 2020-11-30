package me.pqpo.smartcameralib;

import android.graphics.RectF;
import android.view.View;

import com.google.android.cameraview.base.Size;

/**
 * Created by pqpo on 2018/8/20.
 */
public interface  MaskViewImpl {

    View getMaskView();
    RectF getMaskRect();

    void setPreviewSize(Size size);
}
