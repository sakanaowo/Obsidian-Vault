# Refactor R3A — Tạo `ModeSelectorPopup`

Sau R2B, `MainActivity` vẫn còn toàn bộ logic của Mode Popup:

```text
showModePopup()
updateModeCheckmarks()
setupModePopupActions()
selectCalculatorMode()
isCurrentMode()
PopupWindow field
```

R3A sẽ chuẩn bị tách nhóm này sang một class riêng.

Ở bước này:

```text
Tạo 1 file mới
Không sửa MainActivity
Không xóa Mode Popup code cũ
Không thay đổi hành vi app
```

File mới chưa được nối vào `MainActivity`, vì vậy app vẫn chạy bằng implementation Mode Popup hiện tại.

---

# 1. Tạo file

Đường dẫn:

```text
app/src/main/java/com/example/calculator/ui/
ModeSelectorPopup.java
```

Package:

```java
package com.example.calculator.ui;
```

Thay toàn bộ file bằng:

```java
package com.example.calculator.ui;

import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.PopupWindow;

import com.example.calculator.R;
import com.example.calculator.engine.CalculatorMode;

/**
 * Quản lý popup chọn Calculator Mode.
 *
 * Trách nhiệm:
 *
 * - Inflate popup_calculator_mode.xml
 * - Hiển thị PopupWindow bên dưới btnMode
 * - Cập nhật checkmark theo mode hiện tại
 * - Nhận click Basic / Scientific / Convert
 * - Gửi mode được chọn về MainActivity qua callback
 * - Đóng popup và giải phóng reference
 *
 * Không chịu trách nhiệm:
 *
 * - setVisibility() cho mode layouts
 * - lưu currentMode
 * - điều khiển History
 * - điều khiển calculator controller
 */
public final class ModeSelectorPopup {

    // =========================================================
    // CALLBACK
    // =========================================================

    /**
     * MainActivity nhận mode được user chọn.
     */
    public interface OnModeSelectedListener {

        void onModeSelected(
                CalculatorMode mode
        );
    }


    // =========================================================
    // DEPENDENCIES
    // =========================================================

    private final Context context;

    private final OnModeSelectedListener
            modeSelectedListener;


    // =========================================================
    // POPUP STATE
    // =========================================================

    private PopupWindow popupWindow;


    // =========================================================
    // CONSTRUCTOR
    // =========================================================

    public ModeSelectorPopup(
            Context context,
            OnModeSelectedListener modeSelectedListener
    ) {

        if (context == null) {

            throw new IllegalArgumentException(
                    "Context cannot be null."
            );
        }


        if (modeSelectedListener == null) {

            throw new IllegalArgumentException(
                    "Mode listener cannot be null."
            );
        }


        this.context =
                context;


        this.modeSelectedListener =
                modeSelectedListener;
    }


    // =========================================================
    // PUBLIC API
    // =========================================================

    /**
     * Hiển thị popup bên dưới anchor.
     *
     * Nếu popup đang mở, method này sẽ đóng popup hiện tại
     * thay vì tạo popup mới chồng lên.
     */
    public void show(
            View anchor,
            CalculatorMode currentMode
    ) {

        if (anchor == null) {

            return;
        }


        /*
         * Tap btnMode lần nữa khi popup đang mở:
         *
         * → đóng popup.
         */
        if (isShowing()) {

            dismiss();

            return;
        }


        CalculatorMode safeMode =

                currentMode == null

                        ? CalculatorMode.BASIC
                        : currentMode;


        View popupView =
                LayoutInflater
                        .from(context)
                        .inflate(
                                R.layout.popup_calculator_mode,
                                null,
                                false
                        );


        popupWindow =
                createPopupWindow(
                        popupView
                );


        updateCheckmarks(

                popupView,

                safeMode
        );


        setupPopupActions(
                popupView
        );


        popupWindow.setOnDismissListener(
                () -> popupWindow = null
        );


        showAlignedToAnchor(

                anchor,

                popupView
        );
    }


    /**
     * true khi PopupWindow hiện đang hiển thị.
     */
    public boolean isShowing() {

        return popupWindow != null
                && popupWindow.isShowing();
    }


    /**
     * Đóng popup nếu đang tồn tại.
     */
    public void dismiss() {

        if (popupWindow != null) {

            popupWindow.dismiss();
        }
    }


    // =========================================================
    // CREATE POPUP
    // =========================================================

    private PopupWindow createPopupWindow(
            View popupView
    ) {

        PopupWindow createdPopup =
                new PopupWindow(

                        popupView,

                        ViewGroup.LayoutParams.WRAP_CONTENT,

                        ViewGroup.LayoutParams.WRAP_CONTENT,

                        true
                );


        /*
         * Background thật nằm trong:
         *
         * bg_mode_popup.xml
         *
         * PopupWindow dùng transparent background để:
         *
         * - giữ góc bo của XML;
         * - nhận outside touch;
         * - đóng khi tap ra ngoài.
         */
        createdPopup.setBackgroundDrawable(

                new ColorDrawable(
                        Color.TRANSPARENT
                )
        );


        createdPopup.setOutsideTouchable(
                true
        );


        createdPopup.setElevation(

                context
                        .getResources()
                        .getDimension(
                                R.dimen.spacing_8
                        )
        );


        return createdPopup;
    }


    // =========================================================
    // CHECKMARKS
    // =========================================================

    /**
     * Chỉ hiển thị đúng một checkmark
     * tương ứng với currentMode.
     */
    private void updateCheckmarks(
            View popupView,
            CalculatorMode currentMode
    ) {

        ImageView checkBasic =
                popupView.findViewById(
                        R.id.checkModeBasic
                );


        ImageView checkScientific =
                popupView.findViewById(
                        R.id.checkModeScientific
                );


        ImageView checkConvert =
                popupView.findViewById(
                        R.id.checkModeConvert
                );


        checkBasic.setVisibility(

                currentMode == CalculatorMode.BASIC

                        ? View.VISIBLE
                        : View.GONE
        );


        checkScientific.setVisibility(

                currentMode == CalculatorMode.SCIENTIFIC

                        ? View.VISIBLE
                        : View.GONE
        );


        checkConvert.setVisibility(

                currentMode == CalculatorMode.CONVERT

                        ? View.VISIBLE
                        : View.GONE
        );
    }


    // =========================================================
    // ACTIONS
    // =========================================================

    private void setupPopupActions(
            View popupView
    ) {

        popupView
                .findViewById(
                        R.id.rowModeBasic
                )
                .setOnClickListener(view -> {

                    selectMode(
                            CalculatorMode.BASIC
                    );
                });


        popupView
                .findViewById(
                        R.id.rowModeScientific
                )
                .setOnClickListener(view -> {

                    selectMode(
                            CalculatorMode.SCIENTIFIC
                    );
                });


        popupView
                .findViewById(
                        R.id.rowModeConvert
                )
                .setOnClickListener(view -> {

                    selectMode(
                            CalculatorMode.CONVERT
                    );
                });
    }


    private void selectMode(
            CalculatorMode mode
    ) {

        if (mode == null) {

            return;
        }


        /*
         * Callback trước khi đóng popup.
         *
         * MainActivity sẽ:
         *
         * currentMode = mode
         * renderCalculatorMode()
         */
        modeSelectedListener.onModeSelected(
                mode
        );


        dismiss();
    }


    // =========================================================
    // POSITION
    // =========================================================

    /**
     * Căn cạnh phải popup với cạnh phải của btnMode.
     *
     * Điều này tránh popup rộng 280dp bị tràn
     * khỏi cạnh phải màn hình.
     */
    private void showAlignedToAnchor(
            View anchor,
            View popupView
    ) {

        if (popupWindow == null) {

            return;
        }


        popupView.measure(

                View.MeasureSpec.makeMeasureSpec(
                        0,
                        View.MeasureSpec.UNSPECIFIED
                ),

                View.MeasureSpec.makeMeasureSpec(
                        0,
                        View.MeasureSpec.UNSPECIFIED
                )
        );


        int popupWidth =
                popupView.getMeasuredWidth();


        int xOffset =
                anchor.getWidth()
                        - popupWidth;


        int yOffset =
                context
                        .getResources()
                        .getDimensionPixelSize(
                                R.dimen.spacing_8
                        );


        popupWindow.showAsDropDown(

                anchor,

                xOffset,

                yOffset
        );
    }
}
```

---

# 2. Không sửa `MainActivity` ở R3A

Giữ nguyên `MainActivity.java` từ R2B.

Chưa xóa:

```text
PopupWindow modePopupWindow
```

Chưa xóa các import:

```text
android.graphics.Color
android.graphics.drawable.ColorDrawable
android.view.LayoutInflater
android.view.ViewGroup
android.widget.ImageView
android.widget.PopupWindow
```

Chưa xóa các method:

```text
showModePopup()
updateModeCheckmarks()
setupModePopupActions()
selectCalculatorMode()
isCurrentMode()
```

Chưa thay đổi:

```text
setupModeButton()
setCalculatorMode()
renderCalculatorMode()
```

Mode Popup thật của app vẫn chạy từ `MainActivity` cho đến R3B.

---

# 3. Kiến trúc callback

Sau khi nối ở R3B, luồng sẽ là:

```text
btnMode
    ↓
ModeSelectorPopup.show(anchor, currentMode)
    ↓
user chọn Scientific
    ↓
OnModeSelectedListener
    ↓
MainActivity.setCalculatorMode(SCIENTIFIC)
    ↓
renderCalculatorMode()
    ↓
Scientific layout VISIBLE
```

`ModeSelectorPopup` không trực tiếp gọi:

```java
basicCalculatorLayout.setVisibility(...);
```

vì class này không nên biết ba mode root.

Nó chỉ trả về:

```java
CalculatorMode
```

---

# 4. Build checkpoint

Chạy:

```text
Build
→ Make Project
```

Checklist:

```text
[ ] ModeSelectorPopup.java compile

[ ] Package com.example.calculator.ui resolve

[ ] CalculatorMode import resolve

[ ] popup_calculator_mode.xml resolve

[ ] Các ID resolve:
    rowModeBasic
    rowModeScientific
    rowModeConvert

    checkModeBasic
    checkModeScientific
    checkModeConvert

[ ] spacing_8 resolve

[ ] MainActivity chưa thay đổi

[ ] Basic vẫn hoạt động

[ ] Scientific vẫn hoạt động

[ ] History vẫn hoạt động

[ ] Mode popup cũ vẫn hoạt động
```

---

# 5. Lỗi thường gặp

## Lỗi 1 — Không tìm thấy `rowModeConvert`

Thông báo:

```text
Cannot resolve symbol rowModeConvert
```

Kiểm tra:

```text
app/src/main/res/layout/popup_calculator_mode.xml
```

phải có:

```xml
android:id="@+id/rowModeConvert"
```

---

## Lỗi 2 — Không tìm thấy checkmark

Kiểm tra ba ID:

```text
checkModeBasic
checkModeScientific
checkModeConvert
```

phải khớp đúng với `popup_calculator_mode.xml`.

---

## Lỗi 3 — `setElevation()` bị báo API

Project Android hiện tại đã dùng `PopupWindow.setElevation()` trong Bước 8.3.

Nếu Android Studio chỉ hiển thị cảnh báo API nhưng Build vẫn pass, giữ nguyên.

Không thêm dependency ngoài.

---

## Lỗi 4 — Package sai

File phải nằm ở:

```text
com/example/calculator/ui/
ModeSelectorPopup.java
```

và dòng đầu phải là:

```java
package com.example.calculator.ui;
```

Không đặt vào:

```text
ui/controller
```

vì đây là component popup dùng chung, không phải calculator controller.

---

# 6. Trạng thái refactor

```text
[✓] R1A
    Tạo ScientificCalculatorController

[~] R1B
    Nối Scientific controller
    Chờ xác nhận Build/Run

[✓] R2A
    Tạo BasicCalculatorController

[~] R2B
    Nối Basic controller
    Chờ xác nhận Build/Run

[~] R3A
    Tạo ModeSelectorPopup
    Chờ Build

NEXT

R3B
→ nối ModeSelectorPopup vào MainActivity
→ xóa PopupWindow implementation khỏi MainActivity
→ Build
→ test Mode switching
```
