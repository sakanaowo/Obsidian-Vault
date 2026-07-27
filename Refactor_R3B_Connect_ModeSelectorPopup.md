# Refactor R3B — Nối `ModeSelectorPopup` vào `MainActivity`

R3A đã tạo:

```text
app/src/main/java/com/example/calculator/ui/
ModeSelectorPopup.java
```

R3B sẽ:

```text
MainActivity
    ↓
khởi tạo ModeSelectorPopup
    ↓
btnMode gọi popup.show(...)
    ↓
popup trả CalculatorMode qua callback
    ↓
MainActivity.setCalculatorMode(...)
    ↓
renderCalculatorMode()
```

Sau bước này, `MainActivity` không còn trực tiếp:

```text
inflate popup
tạo PopupWindow
căn vị trí popup
cập nhật checkmark
gắn listener cho ba mode row
```

---

# 1. Phạm vi thay đổi

Ở bước này chỉ thay:

```text
app/src/main/java/com/example/calculator/MainActivity.java
```

Không sửa:

```text
ModeSelectorPopup.java

BasicCalculatorController.java
ScientificCalculatorController.java

CalculatorEngine.java
ScientificEngine.java

HistoryManager.java
HistoryBottomSheet.java

activity_main.xml
popup_calculator_mode.xml
```

---

# 2. Thay toàn bộ `MainActivity.java`

File:

```text
app/src/main/java/com/example/calculator/MainActivity.java
```

Thay toàn bộ bằng:

```java
package com.example.calculator;

import android.os.Bundle;
import android.view.View;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.calculator.engine.CalculatorEngine;
import com.example.calculator.engine.CalculatorMode;
import com.example.calculator.engine.ScientificEngine;
import com.example.calculator.history.HistoryManager;
import com.example.calculator.ui.HistoryBottomSheet;
import com.example.calculator.ui.ModeSelectorPopup;
import com.example.calculator.ui.controller.BasicCalculatorController;
import com.example.calculator.ui.controller.ScientificCalculatorController;

/**
 * Main screen của Calculator.
 *
 * MainActivity chỉ chịu trách nhiệm:
 *
 * - Lifecycle
 * - System bars
 * - Bind ba mode root
 * - Khởi tạo các controller
 * - Mở History
 * - Mở Mode Selector
 * - Chuyển Basic / Scientific / Convert
 *
 * Logic của từng phần được tách sang:
 *
 * - BasicCalculatorController
 * - ScientificCalculatorController
 * - ModeSelectorPopup
 * - HistoryBottomSheet
 */
public class MainActivity
        extends AppCompatActivity {

    // =========================================================
    // CONSTANTS
    // =========================================================

    private static final String TAG_HISTORY_BOTTOM_SHEET =
            "HistoryBottomSheet";


    // =========================================================
    // SHARED DATA
    // =========================================================

    private HistoryManager historyManager;


    // =========================================================
    // CONTROLLERS
    // =========================================================

    private BasicCalculatorController
            basicCalculatorController;


    private ScientificCalculatorController
            scientificCalculatorController;


    // =========================================================
    // MODE
    // =========================================================

    private CalculatorMode currentMode;


    private ModeSelectorPopup modeSelectorPopup;


    // =========================================================
    // MODE ROOT VIEWS
    // =========================================================

    private View basicCalculatorLayout;


    private View scientificCalculatorLayout;


    private View converterLayout;


    // =========================================================
    // LIFECYCLE
    // =========================================================

    @Override
    protected void onCreate(
            Bundle savedInstanceState
    ) {

        super.onCreate(
                savedInstanceState
        );


        EdgeToEdge.enable(this);


        setContentView(
                R.layout.activity_main
        );


        setupSystemBars();


        initializeSharedData();


        bindModeLayouts();


        setupControllers();


        setupModeSelector();


        setupHistoryButton();


        setupModeButton();


        renderCalculatorMode();
    }


    @Override
    protected void onDestroy() {

        /*
         * Tránh giữ PopupWindow sau khi Activity bị hủy.
         */
        if (modeSelectorPopup != null) {

            modeSelectorPopup.dismiss();
        }


        super.onDestroy();
    }


    // =========================================================
    // SYSTEM BARS
    // =========================================================

    /**
     * Giữ cách xử lý Edge-to-Edge của Empty Views Activity.
     */
    private void setupSystemBars() {

        ViewCompat.setOnApplyWindowInsetsListener(

                findViewById(
                        R.id.main
                ),

                (view, insets) -> {

                    Insets systemBars =
                            insets.getInsets(
                                    WindowInsetsCompat
                                            .Type
                                            .systemBars()
                            );


                    view.setPadding(

                            systemBars.left,

                            systemBars.top,

                            systemBars.right,

                            systemBars.bottom
                    );


                    return insets;
                }
        );
    }


    // =========================================================
    // INITIALIZATION
    // =========================================================

    private void initializeSharedData() {

        historyManager =
                new HistoryManager(
                        getApplicationContext()
                );


        currentMode =
                CalculatorMode.BASIC;
    }


    // =========================================================
    // MODE ROOT BINDING
    // =========================================================

    private void bindModeLayouts() {

        basicCalculatorLayout =
                findViewById(
                        R.id.basicCalculatorLayout
                );


        scientificCalculatorLayout =
                findViewById(
                        R.id.scientificCalculatorLayout
                );


        converterLayout =
                findViewById(
                        R.id.converterLayout
                );
    }


    // =========================================================
    // CONTROLLERS
    // =========================================================

    private void setupControllers() {

        setupBasicController();


        setupScientificController();
    }


    private void setupBasicController() {

        basicCalculatorController =
                new BasicCalculatorController(

                        basicCalculatorLayout,

                        new CalculatorEngine(),

                        historyManager
                );


        basicCalculatorController.setup();
    }


    private void setupScientificController() {

        scientificCalculatorController =
                new ScientificCalculatorController(

                        scientificCalculatorLayout,

                        new ScientificEngine(),

                        historyManager
                );


        scientificCalculatorController.setup();
    }


    // =========================================================
    // MODE SELECTOR
    // =========================================================

    private void setupModeSelector() {

        modeSelectorPopup =
                new ModeSelectorPopup(

                        this,

                        this::setCalculatorMode
                );
    }


    // =========================================================
    // HISTORY
    // =========================================================

    private void setupHistoryButton() {

        findViewById(
                R.id.btnHistory
        ).setOnClickListener(view -> {

            /*
             * Không mở nhiều History Bottom Sheet chồng nhau.
             */
            if (getSupportFragmentManager()
                    .findFragmentByTag(
                            TAG_HISTORY_BOTTOM_SHEET
                    ) != null) {

                return;
            }


            HistoryBottomSheet historyBottomSheet =
                    new HistoryBottomSheet();


            historyBottomSheet.show(

                    getSupportFragmentManager(),

                    TAG_HISTORY_BOTTOM_SHEET
            );
        });
    }


    // =========================================================
    // MODE BUTTON
    // =========================================================

    private void setupModeButton() {

        findViewById(
                R.id.btnMode
        ).setOnClickListener(view -> {

            modeSelectorPopup.show(

                    view,

                    currentMode
            );
        });
    }


    // =========================================================
    // MODE SWITCHING
    // =========================================================

    /**
     * Được gọi từ callback của ModeSelectorPopup.
     */
    private void setCalculatorMode(
            CalculatorMode mode
    ) {

        if (mode == null) {

            return;
        }


        currentMode =
                mode;


        renderCalculatorMode();
    }


    private void renderCalculatorMode() {

        basicCalculatorLayout.setVisibility(

                currentMode == CalculatorMode.BASIC

                        ? View.VISIBLE
                        : View.GONE
        );


        scientificCalculatorLayout.setVisibility(

                currentMode == CalculatorMode.SCIENTIFIC

                        ? View.VISIBLE
                        : View.GONE
        );


        converterLayout.setVisibility(

                currentMode == CalculatorMode.CONVERT

                        ? View.VISIBLE
                        : View.GONE
        );
    }
}
```

---

# 3. Các phần Mode Popup đã bị loại khỏi `MainActivity`

Sau khi thay toàn bộ file, `MainActivity` không còn field:

```text
PopupWindow modePopupWindow
```

Không còn các import:

```text
android.graphics.Color
android.graphics.drawable.ColorDrawable
android.view.LayoutInflater
android.view.ViewGroup
android.widget.ImageView
android.widget.PopupWindow
```

Không còn các method:

```text
showModePopup()

updateModeCheckmarks()

setupModePopupActions()

selectCalculatorMode()

isCurrentMode()
```

Thay vào đó chỉ còn:

```java
private ModeSelectorPopup modeSelectorPopup;
```

Khởi tạo:

```java
private void setupModeSelector() {

    modeSelectorPopup =
            new ModeSelectorPopup(
                    this,
                    this::setCalculatorMode
            );
}
```

Nút Mode:

```java
private void setupModeButton() {

    findViewById(
            R.id.btnMode
    ).setOnClickListener(view -> {

        modeSelectorPopup.show(
                view,
                currentMode
        );
    });
}
```

---

# 4. Luồng callback sau R3B

```text
User tap btnMode
        ↓
MainActivity.setupModeButton()
        ↓
ModeSelectorPopup.show(view, currentMode)
        ↓
Popup hiển thị đúng checkmark
        ↓
User chọn Scientific
        ↓
ModeSelectorPopup.selectMode(SCIENTIFIC)
        ↓
OnModeSelectedListener
        ↓
MainActivity.setCalculatorMode(SCIENTIFIC)
        ↓
renderCalculatorMode()
        ↓
Scientific VISIBLE
Basic GONE
Convert GONE
        ↓
Popup dismiss
```

`ModeSelectorPopup` không biết:

```text
basicCalculatorLayout
scientificCalculatorLayout
converterLayout
```

`MainActivity` vẫn là nơi duy nhất quản lý ba mode root.

---

# 5. Cấu trúc `MainActivity` sau R3B

```text
MainActivity
│
├── onCreate()
├── onDestroy()
│
├── setupSystemBars()
│
├── initializeSharedData()
├── bindModeLayouts()
│
├── setupControllers()
│   ├── setupBasicController()
│   └── setupScientificController()
│
├── setupModeSelector()
│
├── setupHistoryButton()
├── setupModeButton()
│
├── setCalculatorMode()
└── renderCalculatorMode()
```

Đây là cấu trúc mục tiêu ban đầu của đợt refactor:

```text
MainActivity
= lifecycle + high-level coordination
```

---

# 6. Build checkpoint

Chạy:

```text
Build
→ Make Project
```

Checklist:

```text
[ ] MainActivity.java compile

[ ] ModeSelectorPopup import resolve

[ ] BasicCalculatorController import resolve

[ ] ScientificCalculatorController import resolve

[ ] Không còn import PopupWindow trong MainActivity

[ ] Không còn unresolved method:
    showModePopup
    updateModeCheckmarks
    setupModePopupActions
    selectCalculatorMode
    isCurrentMode

[ ] setupModeSelector() được gọi trước setupModeButton()

[ ] setCalculatorMode() phù hợp với callback
    OnModeSelectedListener

[ ] History compile

[ ] Basic compile

[ ] Scientific compile
```

---

# 7. Lỗi thường gặp

## Lỗi 1 — Callback không khớp

Thông báo có thể là:

```text
Invalid method reference
this::setCalculatorMode
```

Kiểm tra interface trong `ModeSelectorPopup`:

```java
public interface OnModeSelectedListener {

    void onModeSelected(
            CalculatorMode mode
    );
}
```

Và method trong `MainActivity`:

```java
private void setCalculatorMode(
        CalculatorMode mode
)
```

Hai signature phải cùng nhận:

```text
CalculatorMode
```

và trả về:

```text
void
```

---

## Lỗi 2 — `modeSelectorPopup` bị null

Thứ tự trong `onCreate()` phải là:

```text
setupModeSelector()
        ↓
setupModeButton()
```

Không đảo ngược.

---

## Lỗi 3 — Không tìm thấy `ModeSelectorPopup`

Kiểm tra file:

```text
app/src/main/java/com/example/calculator/ui/
ModeSelectorPopup.java
```

Dòng package:

```java
package com.example.calculator.ui;
```

Import:

```java
import com.example.calculator.ui.ModeSelectorPopup;
```

---

## Lỗi 4 — Popup không đóng khi tap lại Mode

`ModeSelectorPopup.show()` phải có:

```java
if (isShowing()) {

    dismiss();

    return;
}
```

Không xóa đoạn này.

---

## Lỗi 5 — Popup checkmark sai

`MainActivity` phải truyền:

```java
currentMode
```

vào:

```java
modeSelectorPopup.show(
        view,
        currentMode
);
```

Không hard-code:

```java
CalculatorMode.BASIC
```

---

# 8. Runtime regression test

Sau khi Build pass, chạy app.

## Test A — Mở/đóng popup

```text
Tap Mode
Expected:
Popup mở
```

Tap lại:

```text
Tap Mode
Expected:
Popup đóng
```

Tap ngoài popup:

```text
Expected:
Popup đóng
```

---

## Test B — Checkmark

Khi app mở:

```text
✓ Basic
  Scientific
  Convert
```

Chọn Scientific, mở lại popup:

```text
  Basic
✓ Scientific
  Convert
```

Chọn Convert, mở lại popup:

```text
  Basic
  Scientific
✓ Convert
```

Luôn chỉ có một checkmark.

---

## Test C — Mode switching

Chuyển:

```text
Basic
→ Scientific
→ Convert
→ Basic
```

Expected:

```text
Không crash
Top bar luôn hiển thị
Mỗi thời điểm chỉ một mode VISIBLE
```

---

## Test D — Basic regression

```text
2 + 3 =
Expected: 5
```

```text
25 × 4 =
Expected: 100
```

Mở History:

```text
Expected:
25 × 4
100
```

---

## Test E — Scientific regression

```text
16
√x

Expected:
4
```

```text
2
xʸ
3
=

Expected:
8
```

```text
30
sin

DEG Expected:
0.5
```

---

## Test F — State preservation

Nhập ở Basic:

```text
25
```

Chuyển sang Scientific, nhập:

```text
16
```

Quay lại Basic:

```text
Expected:
25 vẫn còn
```

Quay lại Scientific:

```text
Expected:
16 vẫn còn
```

Mode switching chỉ thay `VISIBLE/GONE`, không tạo lại controller.

---

# 9. Kết quả sau R3B

```text
MainActivity
│
├── Lifecycle
├── System bars
├── Bind mode roots
├── Initialize controllers
├── History launcher
└── Mode switching


BasicCalculatorController
│
├── Basic listeners
├── Basic render
└── Basic History


ScientificCalculatorController
│
├── Scientific listeners
├── Scientific state/render
└── Scientific History


ModeSelectorPopup
│
├── PopupWindow
├── Checkmarks
├── Mode row listeners
└── Mode callback
```

Trạng thái refactor:

```text
[✓] R1A
    Tạo ScientificCalculatorController

[✓] R1B
    Nối Scientific controller

[✓] R2A
    Tạo BasicCalculatorController

[✓] R2B
    Nối Basic controller

[✓] R3A
    Tạo ModeSelectorPopup

[~] R3B
    Nối ModeSelectorPopup
    Chờ Build + Runtime test
```

Sau khi R3B Build và Run pass, đợt refactor `MainActivity` hoàn tất.

Bước tiếp theo theo roadmap:

```text
8.8
Scientific Functional Test
+
sửa edge case nếu có

sau đó mới chuyển sang:
Converter
```
